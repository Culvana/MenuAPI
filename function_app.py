import os
import json
import logging
import azure.functions as func
import azure.durable_functions as df
from azure.storage.blob.aio import BlobServiceClient
from azure.cosmos.aio import CosmosClient
from azure.servicebus.aio import ServiceBusClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.search.documents import SearchClient
from openai import OpenAI
from typing import List, Dict, Any
import mimetypes
import tempfile
from datetime import datetime
from azure.servicebus import ServiceBusMessage
import asyncio

from shared.recipe_parser import process_recipe_folder, RecipeCostCalculator

# Initialize function app
myapp = df.DFApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Initialize blob client
blob_service_client = BlobServiceClient.from_connection_string(os.environ["AzureWebJobsStorage"])

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def get_or_create_container(user_id: str):
    """Get or create a container for the user."""
    container_name = f"user{user_id}recipes"
    container_client = blob_service_client.get_container_client(container_name)
    
    try:
        await container_client.create_container()
        logging.info(f"Created new container: {container_name}")
    except Exception as e:
        if "ContainerAlreadyExists" in str(e):
            logging.info(f"Container already exists: {container_name}")
        else:
            logging.error(f"Error creating container: {str(e)}")
            raise
    
    return container_client, container_name

def is_valid_file_type(filename: str) -> bool:
    """Check if file type is allowed."""
    allowed_extensions = {'.pdf', '.xlsx', '.xls', '.csv', '.jpg', '.jpeg', '.png'}
    return os.path.splitext(filename)[1].lower() in allowed_extensions
# Add near the top of the file, after imports
async def validate_azure_environment():
    """Validate Azure Functions environment."""
    try:
        required_vars = [
            "AzureWebJobsStorage",
            "AZURE_AISEARCH_ENDPOINT",
            "AZURE_AISEARCH_APIKEY",
            "AZURE_FORM_RECOGNIZER_ENDPOINT",
            "AZURE_FORM_RECOGNIZER_KEY",
            "COSMOS_ENDPOINT",
            "COSMOS_KEY",
            "OPENAI_API_KEY",
            "ServiceBusConnection"
        ]
        
        missing = [var for var in required_vars if not os.environ.get(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
            
        logging.info("Azure environment validated successfully")
        
    except Exception as e:
        logging.error(f"Azure environment validation failed: {str(e)}")
        raise

async def startup():
    await validate_azure_environment()

@myapp.route(route="process-menu/{user_id}/{index_name}", methods=["POST"])
@myapp.durable_client_input(client_name="client")
async def http_trigger(req: func.HttpRequest, client):
    """HTTP trigger for recipe processing."""
    try:
        user_id = req.route_params.get('user_id')
        index_name = req.route_params.get('index_name')

        if not user_id:
            return func.HttpResponse(
                json.dumps({"error": "User ID is required"}),
                mimetype="application/json",
                status_code=400
            )
        if not index_name:
            return func.HttpResponse(
                json.dumps({"error": "Index name is required"}),
                mimetype="application/json",
                status_code=400
            )

        # Get or create the container
        container_client, container_name = await get_or_create_container(user_id)

        # Handle file uploads
        blob_references = []
        max_file_size = 50 * 1024 * 1024  # 50MB limit
        
        for file_name in req.files:
            file = req.files[file_name]
            
            if not is_valid_file_type(file.filename):
                return func.HttpResponse(
                    json.dumps({"error": f"Invalid file type: {file.filename}"}),
                    mimetype="application/json",
                    status_code=400
                )

            file_content = file.read()
            if len(file_content) > max_file_size:
                return func.HttpResponse(
                    json.dumps({"error": f"File too large: {file.filename}"}),
                    mimetype="application/json",
                    status_code=400
                )
            file.seek(0)

            # Use sanitized filename for blob
            safe_filename = ''.join(c for c in file.filename if c.isalnum() or c in '._-')
            blob_name = f"{index_name}/{safe_filename}"
            
            # Create blob client using the existing container
            blob_client = container_client.get_blob_client(blob_name)
            
            # Upload the file
            await blob_client.upload_blob(file.stream, overwrite=True)
            blob_references.append({
                "blob_name": blob_name,
                "container_name": container_name
            })

        if not blob_references:
            return func.HttpResponse(
                json.dumps({"error": "No files uploaded"}),
                mimetype="application/json",
                status_code=400
            )

        # Start orchestration
        instance_id = await client.start_new(
            "recipe_orchestrator",
            None,
            {
                "user_id": user_id,
                "index_name": index_name,
                "blobs": blob_references
            }
        )
        
        return client.create_check_status_response(req, instance_id)

    except Exception as e:
        logging.error(f"Error in http_trigger: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": "Internal server error", "details": str(e)}),
            mimetype="application/json",
            status_code=500
        )

@myapp.orchestration_trigger(context_name="context")
def recipe_orchestrator(context:df.DurableOrchestrationContext):
    """Orchestrator function for recipe processing."""
    try:
        input_data = context.get_input()
        if isinstance(input_data, str):
            input_data = json.loads(input_data)
            
        user_id = input_data.get("user_id")
        index_name = input_data.get("index_name")
        blobs = input_data.get("blobs", [])

        if not user_id or not blobs or not index_name:
            return {
                "status": "failed",
                "message": "Invalid input data",
                "recipe_count": 0
            }

        # Process files in parallel
        tasks = []
        for blob in blobs:
            task = context.call_activity("process_file_activity", {
                "blob": blob,
                "index_name": index_name,
                "user_id": user_id
            })
            tasks.append(task)

        results = yield context.task_all(tasks)
        valid_results = [r for result in results for r in result if r and isinstance(r, dict)]
        
        if not valid_results:
            return {
                "status": "completed",
                "message": "No valid recipes found",
                "recipe_count": 0
            }

        # Store results
        store_result = yield context.call_activity("store_recipes_activity", {
            "user_id": user_id,
            "index_name": index_name,
            "recipes": valid_results
        })

        return {
            "status": "completed",
            "message": f"Processed {len(valid_results)} recipes",
            "recipe_count": len(valid_results),
            "total_user_recipes": store_result.get("total_user_recipes", 0)
        }

    except Exception as e:
        return {"status": "failed", "error": str(e)}

@myapp.activity_trigger(input_name="taskinput")
async def process_file_activity(taskinput):
    """Activity function for processing individual files."""
    temp_dir = None
    try:
        # Validate input
        if not isinstance(taskinput, dict):
            logging.error(f"Invalid taskinput type: {type(taskinput)}")
            return []
            
        blob_info = taskinput.get("blob")
        index_name = taskinput.get("index_name")
        user_id = taskinput.get("user_id")

        logging.info(f"Starting process_file_activity with: blob={blob_info}, index={index_name}, user={user_id}")

        if not all([blob_info, index_name, user_id]):
            logging.error(f"Missing required fields: blob_info={bool(blob_info)}, index_name={bool(index_name)}, user_id={bool(user_id)}")
            return []

        try:
            # Initialize clients
            search_client = SearchClient(
                endpoint=os.environ["AZURE_AISEARCH_ENDPOINT"],
                index_name=index_name,
                credential=AzureKeyCredential(os.environ["AZURE_AISEARCH_APIKEY"])
            )
            logging.info("Search client initialized")
            
            form_client = DocumentAnalysisClient(
                endpoint=os.environ["AZURE_FORM_RECOGNIZER_ENDPOINT"],
                credential=AzureKeyCredential(os.environ["AZURE_FORM_RECOGNIZER_KEY"])
            )
            logging.info("Form client initialized")
            
            openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            logging.info("OpenAI client initialized")

            # Create base temp directory
            base_temp_dir = tempfile.mkdtemp(prefix='recipe_processing_')
            logging.info(f"Created base temp directory at {base_temp_dir}")

            # Create specific directory for this file
            recipe_dir = os.path.join(base_temp_dir, 'recipe_files')
            os.makedirs(recipe_dir)
            logging.info(f"Created recipe directory at {recipe_dir}")

            # Download blob with retries
            max_retries = 3
            file_path = None
            for attempt in range(max_retries):
                try:
                    container_client = blob_service_client.get_container_client(blob_info["container_name"])
                    blob_client = container_client.get_blob_client(blob_info["blob_name"])
                    download_stream = await blob_client.download_blob()
                    
                    file_name = os.path.basename(blob_info["blob_name"])
                    file_path = os.path.join(recipe_dir, file_name)
                    
                    with open(file_path, "wb") as temp_file:
                        async for chunk in download_stream.chunks():
                            temp_file.write(chunk)
                    
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        logging.info(f"Successfully downloaded blob to {file_path} (size: {os.path.getsize(file_path)} bytes)")
                        break
                    else:
                        raise Exception("Downloaded file is empty or missing")
                        
                except Exception as e:
                    if attempt == max_retries - 1:
                        logging.error(f"Failed to download blob after {max_retries} attempts: {str(e)}")
                        return []
                    logging.warning(f"Retry {attempt + 1} downloading blob: {str(e)}")
                    await asyncio.sleep(1)

            if not file_path or not os.path.exists(file_path):
                logging.error("Failed to download file")
                return []

            # Log directory contents and file info
            logging.info(f"Directory contents: {os.listdir(recipe_dir)}")
            logging.info(f"File exists: {os.path.exists(file_path)}")
            logging.info(f"File size: {os.path.getsize(file_path)}")

            # Initialize calculator
            calculator = RecipeCostCalculator(search_client, openai_client, user_id)
            logging.info("Calculator initialized")

            # Process recipes
            try:
                logging.info(f"Starting recipe processing for directory: {recipe_dir}")
                
                recipes = await process_recipe_folder(
                    recipe_dir,  # Pass the directory containing the file
                    calculator,
                    form_client,
                    openai_client
                )
                
                logging.info(f"Recipe processing result type: {type(recipes)}")
                if recipes is not None:
                    logging.info(f"Recipes found: {len(recipes) if isinstance(recipes, list) else 'Not a list'}")
                
                if recipes is None:
                    logging.error("process_recipe_folder returned None")
                    return []
                    
                if not isinstance(recipes, list):
                    logging.error(f"process_recipe_folder returned unexpected type: {type(recipes)}")
                    return []
                    
                if not recipes:
                    logging.warning("No recipes found in file")
                    return []

                logging.info(f"Successfully processed {len(recipes)} recipes")
                return recipes

            except Exception as e:
                logging.error(f"Error in process_recipe_folder: {str(e)}")
                logging.error(f"Error details: {str(e.__class__.__name__)}")
                return []

        except Exception as e:
            logging.error(f"Error processing file: {str(e)}")
            return []

    except Exception as e:
        logging.error(f"Error in process_file_activity: {str(e)}")
        return []
        
    finally:
        if base_temp_dir and os.path.exists(base_temp_dir):
            try:
                import shutil
                shutil.rmtree(base_temp_dir)
                logging.info(f"Cleaned up temporary directory: {base_temp_dir}")
            except Exception as e:
                logging.error(f"Error cleaning up temp directory: {str(e)}")

@myapp.activity_trigger(input_name="storeinput")
async def store_recipes_activity(storeinput: Dict[str, Any]) -> Dict:
    """Activity function for storing processed recipes."""
    try:
        user_id = storeinput.get("user_id")
        recipes = storeinput.get("recipes", [])
        index_name = storeinput.get("index_name")

        if not user_id or not index_name:
            raise ValueError("User ID and index name are required")

        if not recipes:
            return {
                "status": "completed",
                "message": "No recipes to store",
                "stored_count": 0,
                "total_user_recipes": 0
            }

        async with CosmosClient(
            url=os.environ["COSMOS_ENDPOINT"],
            credential=os.environ["COSMOS_KEY"]
        ) as cosmos_client:
            database = cosmos_client.get_database_client("InvoicesDB")
            container = database.get_container_client("Menu")

            # Get current user document or create new one
            try:
                user_doc = await container.read_item(
                    item=user_id,
                    partition_key=user_id
                )
            except:
                user_doc = {
                    "id": user_id,
                    "type": "user",
                    "recipe_count": 0,
                    "recipes": {}
                }

            # Update recipe count and add new recipes
            current_count = user_doc.get("recipe_count", 0)
            
            # Organize recipes by index
            if index_name not in user_doc["recipes"]:
                user_doc["recipes"][index_name] = []

            # Add new recipes with sequential numbering
            for i, recipe in enumerate(recipes, start=current_count + 1):
                recipe_entry = {
                    "id": f"{user_id}_{index_name}_{i}",
                    "sequence_number": i,
                    "name": recipe["recipe_name"],
                    "created_at": datetime.utcnow().isoformat(),
                    "data": recipe
                }
                user_doc["recipes"][index_name].append(recipe_entry)

            # Update total recipe count and timestamp
            user_doc["recipe_count"] = current_count + len(recipes)
            user_doc["last_updated"] = datetime.utcnow().isoformat()

            # Save updated user document
            await container.upsert_item(user_doc)

            # Send notification to Service Bus
            async with ServiceBusClient.from_connection_string(
                os.environ["ServiceBusConnection"]
            ) as servicebus_client:
                sender = servicebus_client.get_queue_sender("recipe-updates")
                message = {
                    "user_id": user_id,
                    "index_name": index_name,
                    "new_recipes": len(recipes),
                    "total_recipes": user_doc["recipe_count"],
                    "status": "completed"
                }
                await sender.send_messages([
                    ServiceBusMessage(json.dumps(message))
                ])

            return {
                "status": "completed",
                "message": f"Successfully stored {len(recipes)} recipes",
                "stored_count": len(recipes),
                "total_user_recipes": user_doc["recipe_count"]
            }

    except Exception as e:
        logging.error(f"Error storing recipes: {str(e)}")
        raise