# data_agent.py
from python_a2a import A2AServer, Message, TextContent, MessageRole, run_server
from python_a2a.mcp import A2AMCPAgent, MCPClient
from python_a2a import agent, skill
from typing import Dict, Any, List, Optional
import asyncio
import json
import traceback
import httpx
from urllib.parse import urljoin
import os

def load_validation_rules() -> Dict[str, Any]:
    """Load validation rules from the JSON file"""
    try:
        # Get the path to the validation rules file
        config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "schemas")
        rules_path = os.path.join(config_dir, "omop_validation_rules.json")
        
        print(f"Loading validation rules from: {rules_path}")
        
        with open(rules_path, "r") as f:
            rules = json.load(f)
            print(f"Successfully loaded validation rules with {len(rules.get('required_tables', []))} required tables")
            return rules
    except Exception as e:
        print(f"Error loading validation rules: {e}")
        traceback.print_exc()
        # Return a minimal set of rules if loading fails
        return {
            "required_tables": [],
            "required_joins": [],
            "required_columns": []
        }
@agent(
    name="EhrRetrievalAgent",
    description="Agent that provides OMOP CDM retrieved data.",
    version="0.1.0"
)
class EhrRetrievalAgent(A2AServer, A2AMCPAgent):
    """Agent that provides OMOP CDM retrieved data."""
    
    def __init__(self):
        # Initialize A2AServer (first parent)
        A2AServer.__init__(
            self,
            name="EhrRetrievalAgent",
            description="Agent that provides OMOP CDM retrieved data.",
            version="0.1.0"
        )
        
        # Create and set the event loop first
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        # Initialize A2AMCPAgent (second parent) with MCP server URL
        A2AMCPAgent.__init__(
            self,
            name="EhrRetrievalAgent",
            description="Agent that provides OMOP CDM retrieved data.",
            mcp_servers={
                "omop-data": "http://localhost:8000"
            }
        )
        
        # Load validation rules on initialization
        self.validation_rules = load_validation_rules()
        
        # Initialize MCP client with proper timeout settings
        self.mcp_client = MCPClient(
            server_url="http://localhost:8000",
        )


    
    @skill(
        name="Execute_SQL_Query",
        description="Execute a SQL query against an OMOP database and return results",
    )
    async def execute_query(self, query: str) -> Any:
        """Execute a SQL query against an OMOP database and return results"""
        try:
            from python_a2a.models.content import FunctionCallContent, FunctionParameter
            
            # Create function call content
            function_call = FunctionCallContent(
                name="Execute_SQL_Query",
                parameters=[FunctionParameter(name="query", value=query)]
            )
            
            # Process the function call
            result = await self.handle_function_call(function_call)
            return result
        except Exception as e:
            print(f"Error in execute_query: {e}")
            traceback.print_exc()
            return {"error": str(e)}
    
    @skill(
        name="Test_Connection",
        description="Test if a database connection is valid",
    )
    async def test_connection(self, connection_string: str) -> Any:
        """Test if a database connection is valid"""
        try:
            from python_a2a.models.content import FunctionCallContent, FunctionParameter
            
            # Create function call content
            function_call = FunctionCallContent(
                name="Test_Connection",
                parameters=[FunctionParameter(name="connection_string", value=connection_string)]
            )
            
            # Process the function call
            result = await self.handle_function_call(function_call)
            return result
        except Exception as e:
            print(f"Error in test_connection: {e}")
            traceback.print_exc()
            return {"error": str(e)}
    
    @skill(
        name="Get_OMOP_Schema",
        description="Get the OMOP CDM schema information for prompting",
    )
    async def get_omop_schema(self) -> Any:
        """Get the OMOP CDM schema information for prompting"""
        try:
            from python_a2a.models.content import FunctionCallContent
            
            # Create function call content with no parameters
            function_call = FunctionCallContent(
                name="Get_OMOP_Schema",
        parameters=[]
    )
            
            # Process the function call
            result = await self.handle_function_call(function_call)
            return result
        except Exception as e:
            print(f"Error in get_omop_schema: {e}")
            traceback.print_exc()
            return {"error": str(e)}
    
    @skill(
        name="Validate_SQL_Query",
        description="Validate a SQL query against OMOP CDM validation rules",
    )
    async def validate_query(self, sql_query: str) -> Dict[str, Any]:
        """Validate a SQL query against OMOP CDM validation rules"""
        try:
            from python_a2a.models.content import FunctionCallContent, FunctionParameter
            
            # Create function call content
            function_call = FunctionCallContent(
                name="Validate_SQL_Query",
                parameters=[FunctionParameter(name="sql_query", value=sql_query)]
            )
            
            # Process the function call
            result = await self.handle_function_call(function_call)
            
            # Ensure we have a dictionary response
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    return {"is_valid": False, "issues": [f"Invalid validation response: {result}"]}
            
            # Ensure the result has the expected structure
            if not isinstance(result, dict):
                return {"is_valid": False, "issues": [f"Unexpected validation response type: {type(result)}"]}
            
            return result
        except Exception as e:
            print(f"Error in validate_query: {e}")
            traceback.print_exc()
            return {"is_valid": False, "issues": [str(e)]}
    
    @skill(
        name="Generate_SQL",
        description="Generate a SQL query from natural language",
    )
    async def generate_sql(self, prompt: str) -> Any:
        """Generate SQL from natural language using an LLM"""
        try:
            from python_a2a.models.content import FunctionCallContent, FunctionParameter
            
            # First get the schema
            schema = await self.get_omop_schema()
            
            # Format schema as string if needed
            if not isinstance(schema, str):
                schema_str = json.dumps(schema) if isinstance(schema, dict) else str(schema)
            else:
                schema_str = schema
            
            # Create function call content
            function_call = FunctionCallContent(
                name="Generate_SQL",
                parameters=[
                    FunctionParameter(name="prompt", value=prompt),
                    FunctionParameter(name="schema", value=schema_str)
                ]
            )
            
            # Process the function call
            result = await self.handle_function_call(function_call)
            return result
        except Exception as e:
            print(f"Error in generate_sql: {e}")
            traceback.print_exc()
            return {"error": str(e), "sql_query": ""}
    @skill(
        name="Call_Ollama_For_Explanation",
        description="Call Ollama API directly for SQL explanation",
    )
    async def call_ollama_for_explanation(self, sql_query: str, model_name: str = "codellama:latest") -> str:
        """Call Ollama API directly for SQL explanation"""
        ollama_base_url = "http://localhost:11434"
        api_endpoint = "/api/generate"
        
        # Prepare the system prompt and format the full prompt
        system_prompt = "You are a healthcare analytics expert explaining SQL queries to clinicians."
        
        explanation_prompt = f"""System: {system_prompt}

Explain what this SQL query does in simple terms, focusing on the healthcare insights it provides:

```sql
{sql_query}
```

Explain in 2-3 sentences what clinical question this query answers and how it uses the OMOP CDM structure.
Focus on the medical/clinical meaning of the results, not the technical SQL details.
"""
        
        # Prepare the request for Ollama
        ollama_request = {
            "model": model_name,
            "prompt": explanation_prompt,
            "stream": False
        }
        
        try:
            print(f"Calling Ollama for explanation at {ollama_base_url}{api_endpoint}")
            
            # Make async request to Ollama
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    urljoin(ollama_base_url, api_endpoint),
                    json=ollama_request,
                    timeout=60.0
                )
                response.raise_for_status()
                
                # Get the response data
                response_data = response.json()
                print(f"Ollama explanation response: {response_data}")
                
                if "response" in response_data:
                    return response_data["response"].strip()
                else:
                    return "Could not generate explanation"
                
        except Exception as e:
            print(f"Error calling Ollama for explanation: {e}")
            traceback.print_exc()
            return "Error generating explanation"

    @skill(
        name="Generate_Explanation",
        description="Generate an explanation for an SQL query",
    )
    async def generate_explanation(self, sql_query: str) -> Any:
        """Generate an explanation for an SQL query"""
        try:
            # Call Ollama directly for explanation
            explanation = await self.call_ollama_for_explanation(sql_query)
            return {
                "status": "success",
                "explanation": explanation
            }
        except Exception as e:
            print(f"Error in generate_explanation: {e}")
            traceback.print_exc()
            return {
                "status": "error",
                "error": str(e)
            }
    
    @skill(
        name="Call_Ollama_Directly",
        description="Call Ollama API directly for SQL generation",
    )
    async def call_ollama_directly(self, prompt: str, model_name: str = "codellama:latest") -> str:
        """Call Ollama API directly for SQL generation"""
        ollama_base_url = "http://localhost:11434"
        api_endpoint = "/api/generate"
        
        # Prepare the system prompt and format the full prompt
        system_prompt = "You are an expert in SQL and healthcare data analysis, specifically working with the OMOP Common Data Model (CDM)."
        
        try:
            # First get the schema
            schema = await self.get_omop_schema()
            if isinstance(schema, dict) and schema.get("status") == "success":
                schema_text = schema.get("schema_text", "")
            else:
                schema_text = str(schema)
            
            # Load example queries
            try:
                config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "schemas")
                examples_path = os.path.join(config_dir, "examples.json")
                with open(examples_path, "r") as f:
                    examples_data = json.load(f)
                    example_queries = examples_data.get("omop_queries", [])
            except Exception as e:
                print(f"Error loading example queries: {e}")
                example_queries = []
            
            # Select most relevant examples based on query similarity
            def get_relevant_examples(query, examples, max_examples=3):
                # Simple keyword matching for now
                query_lower = query.lower()
                scored_examples = []
                
                for example in examples:
                    score = 0
                    # Check for matching keywords
                    keywords = ['patient', 'count', 'age', 'gender', 'condition', 'drug', 'measurement', 
                              'visit', 'procedure', 'diabetes', 'hypertension', 'heart', 'cancer']
                    
                    for keyword in keywords:
                        if keyword in query_lower and keyword in example['question'].lower():
                            score += 1
                    
                    # Check for matching complexity
                    if 'complex' in query_lower and example['complexity'] in ['complex', 'very complex']:
                        score += 1
                    elif 'simple' in query_lower and example['complexity'] == 'simple':
                        score += 1
                    
                    scored_examples.append((score, example))
                
                # Sort by score (first element of tuple) and get top examples
                scored_examples.sort(key=lambda x: x[0], reverse=True)
                return [ex for _, ex in scored_examples[:max_examples]]
            
            # Get relevant examples
            relevant_examples = get_relevant_examples(prompt, example_queries)
            
            # Format example queries
            examples_text = "\n## Relevant Example Queries:\n"
            for example in relevant_examples:
                examples_text += f"\nQuestion: {example['question']}\n"
                examples_text += f"SQL: {example['sql']}\n"
            
            full_prompt = f"""System: {system_prompt}

Given the following OMOP CDM schema:

{schema_text}

{examples_text}

Convert the following natural language query into a valid SQL query that follows OMOP CDM best practices:

"{prompt}"

## ABOUT OMOP CDM:
The OMOP CDM is a standardized data model for observational healthcare data with these key characteristics:
- Person-centric design
- Standardized vocabularies with concept_id system
- Event tables (condition_occurrence, drug_exposure, procedure_occurrence, measurement, observation)
- Standardized clinical concepts using vocabulary tables

## QUERY GUIDELINES:
1. Always use standard SQL syntax compatible with PostgreSQL
2. Join person table when querying patient-level data
3. When filtering by clinical concepts, join to the concept table
4. Add appropriate date filters when relevant (use condition_start_date, drug_exposure_start_date, etc.)
5. Always properly handle NULL values
6. Use COUNT(DISTINCT person_id) for patient counts
7. Use concept_id filters for clinical entities, not string matching when possible
8. Include relevant JOINs between clinical tables and the person table
9. Use common tables when needed: person, visit_occurrence, condition_occurrence, drug_exposure, procedure_occurrence, measurement, observation, concept

## KEY TABLE RELATIONSHIPS:
- All clinical events link to person via person_id
- All events can link to visit_occurrence via visit_occurrence_id 
- Clinical events link to concepts via _concept_id fields
- Concepts are in the concept table with standard concept_id values

## OUTPUT FORMAT:
Return ONLY the SQL query with no explanation, comments, markdown formatting, or additional text.

Always format your response as a valid SQL query with no additional text.
"""
            
            # Prepare the request for Ollama
            ollama_request = {
                "model": model_name,
                "prompt": full_prompt,
                "stream": False
            }
            
            print(f"Calling Ollama directly at {ollama_base_url}{api_endpoint}")
            
            # Make async request to Ollama
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    urljoin(ollama_base_url, api_endpoint),
                    json=ollama_request,
                    timeout=60.0
                )
                response.raise_for_status()
                
                # Get the response data
                response_data = response.json()
                print(f"Ollama direct response: {response_data}")
                
                # Extract the SQL query from the response
                if "response" in response_data:
                    # Clean up the response to extract just the SQL query
                    full_response = response_data["response"].strip()
                    
                    # Try to extract SQL from markdown code blocks first
                    import re
                    sql_match = re.search(r'```(?:sql)?\s*(.*?)\s*```', full_response, re.DOTALL)
                    if sql_match:
                        sql_query = sql_match.group(1).strip()
                    else:
                        # If no code block, try to find a SELECT statement
                        sql_match = re.search(r'(SELECT\s+.*?;)', full_response, re.DOTALL | re.IGNORECASE)
                        if sql_match:
                            sql_query = sql_match.group(1).strip()
                        else:
                            # If still no SQL found, use the whole response
                            sql_query = full_response
                    
                    return sql_query
                else:
                    return "SELECT COUNT(*) FROM person"  # Fallback query
            
        except Exception as e:
            print(f"Error calling Ollama directly: {e}")
            traceback.print_exc()
            # Return a simple fallback query
            return "SELECT COUNT(*) FROM person"
    
    def handle_message(self, message):
        """Handle incoming messages by routing to async handler"""
        try:
            # Use the instance's event loop
            return self._loop.run_until_complete(self.handle_message_async(message))
        except RuntimeError:
            # If the loop is closed, create a new one
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            return self._loop.run_until_complete(self.handle_message_async(message))
    
    async def handle_function_call(self, function_call):
        """Handle function calls while managing event loop issues"""
        try:
            # Extract function name and parameters
            function_name = function_call.name
            params = {p.name: p.value for p in function_call.parameters}
            
            # Initialize MCP client if not exists
            if self.mcp_client is None:
                print("Initializing MCP client...")
                from python_a2a.mcp.client import MCPClient
                self.mcp_client = MCPClient(
                    server_url="http://localhost:8000",
                    timeout=30.0,
                    max_retries=3
                )
            
            # Try to use our direct MCP client
            try:
                print(f"Calling tool {function_name} using direct MCP client")
                result = await self.mcp_client.call_tool(function_name, **params)
                
                # Validate the result
                if isinstance(result, dict) and result.get("status") == "error":
                    print(f"Error from MCP client: {result.get('error')}")
                    # Try parent class handling as fallback
                    print("Falling back to parent class MCP handling")
                    return await super().handle_function_call(function_call)
                
                return result
                
            except RuntimeError as e:
                if "bound to a different event loop" in str(e):
                    print("Event loop conflict detected, recreating MCP client")
                    # Create a new client on the current event loop
                    from python_a2a.mcp.client import MCPClient
                    self.mcp_client = MCPClient(
                        server_url="http://localhost:8000",
                        timeout=30.0,
                        max_retries=3
                    )
                    return await self.mcp_client.call_tool(function_name, **params)
                else:
                    raise
            
        except Exception as e:
            print(f"Error in handle_function_call: {e}")
            traceback.print_exc()
            # Return an error response
            return {
                "error": f"Failed to call function {function_name}: {str(e)}",
                "status": "error"
            }
    
    async def handle_message_async(self, message: Message) -> Message:
        """Asynchronously handle messages with simplified flow"""
        try:
            if message.role == MessageRole.USER:
                print(f"Received user message: {message.content.text}")
                
                # Step 1: Generate initial SQL query
                sql_query = await self.call_ollama_directly(message.content.text)
                print(f"Initial SQL query: {sql_query}")
                
                # Step 2: Validate the query
                validation_result = await self.validate_query(sql_query)
                print(f"Initial validation result: {validation_result}")
                
                # Step 3: If validation fails, try to refine the query
                if not validation_result.get("is_valid", False):
                    print("Validation issues found, attempting to refine query...")
                    issues = validation_result.get("issues", [])
                    
                    # Create feedback prompt with validation issues
                    feedback_prompt = f"""The following SQL query failed validation:
{sql_query}

Validation issues:
{chr(10).join(f'- {issue}' for issue in issues)}

Please generate a corrected SQL query that addresses these issues.
Remember to:
1. Include all required tables and joins
2. Select required columns
3. Add appropriate date range filters
4. Follow OMOP CDM best practices

Your corrected SQL query:"""
                    
                    # Generate refined query
                    sql_query = await self.call_ollama_directly(feedback_prompt)
                    print(f"Refined SQL query: {sql_query}")
                    
                    # Validate the refined query
                    validation_result = await self.validate_query(sql_query)
                    print(f"Refined validation result: {validation_result}")
                
                # Step 4: Execute the query
                print(f"Executing final SQL query: {sql_query}")
                query_result = await self.execute_query(sql_query)
                
                # Step 5: Generate explanation
                explanation = await self.call_ollama_for_explanation(sql_query)
                
                # Build response
                response_parts = []
                if not validation_result.get("is_valid", False):
                    response_parts.append("Warning: Query has validation issues but will be executed anyway:")
                    for issue in validation_result.get("issues", []):
                        response_parts.append(f"- {issue}")
                    response_parts.append("")
                
                response_parts.append(f"Query results: {query_result}")
                response_parts.append(f"\nExplanation: {explanation}")
                
                response_text = "\n".join(response_parts)
                return Message(role=MessageRole.AGENT, content=TextContent(text=response_text))
            else:
                return Message(
                    role=MessageRole.AGENT,
                    content=TextContent(text="I only process user queries about OMOP data.")
                )
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return Message(role=MessageRole.AGENT, content=TextContent(text=error_msg))