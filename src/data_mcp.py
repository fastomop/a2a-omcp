from python_a2a.mcp import FastMCP
import httpx
import json
import re
import time
import asyncio
import os
from typing import Dict, Any, List, Tuple, Optional
from sqlalchemy import create_engine, text
from datetime import datetime
import sqlglot
from sqlglot import expressions as exp  # This is the missing import
from typing import Dict, Any, List, Set

# Initialize a single MCP server
mcp = FastMCP(name="OMOP Unified MCP Server")

# ============== SHARED UTILITIES ===============

# Internal state for database connections
_db_engines = {}


def get_config():
    """Get application configuration"""
    try:
        from app.core.config import settings
        return settings.config
    except ImportError:
        # Fallback configuration for direct usage
        config_path = os.environ.get("CONFIG_PATH", "config/config.json")
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {
                "database": {
                    "connection_strings": {
                        "default": "postgresql://root:root@localhost:5432/postgres?options=-c%20search_path=omop_cdm"
                    },
                    "schema_directory": "config/schemas"
                },
                "ollama": {
                    "api_url": "http://localhost:11434/api/generate",
                    "default_model": "codellama:latest"
                },
                "omop_cdm": {
                    "validation_rules": "omop_validation_rules.json",
                    "schema_file": "omop_cdm_schema.json"
                }
            }


def get_db_engine(connection_id: Optional[str] = None, connection_string: Optional[str] = None):
    """Get or create a database engine"""
    global _db_engines

    if connection_string:
        # Create a temporary engine for one-time use
        return create_engine(connection_string)

    # Use a connection from the config
    conn_id = connection_id or "default"

    if conn_id not in _db_engines:
        try:
            config = get_config()
            conn_string = config["database"]["connection_strings"].get(conn_id)
            if not conn_string:
                raise ValueError(f"No connection string found for {conn_id}")
            _db_engines[conn_id] = create_engine(conn_string)
        except Exception as e:
            raise Exception(f"Failed to create engine for {conn_id}: {e}")

    return _db_engines[conn_id]


def load_validation_rules() -> Dict[str, Any]:
    """Load validation rules from file"""
    try:
        config = get_config()
        rules_path = os.path.join(
            config["omop_cdm"]["validation_rules"]
        )
        print(f"Loading validation rules from: {rules_path}")  # Debug print
        with open(rules_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading validation rules: {e}")
        return {
            "required_tables": [],
            "required_joins": [],
            "concept_tables": []
        }


# Load validation rules on startup
VALIDATION_RULES = load_validation_rules()


# ============== SQL SERVER TOOLS ===============

@mcp.tool(
    name="Execute_SQL_Query",
    description="Execute a SQL query against an OMOP database and return results as structured JSON"
)
def execute_query(query: str, connection_id: str = None, connection_string: str = None) -> Dict[str, Any]:
    """Execute a SQL query and return results as structured JSON"""
    start_time = time.time()

    try:
        engine = get_db_engine(connection_id, connection_string)

        with engine.connect() as connection:
            result = connection.execute(text(query))
            column_names = result.keys()
            
            # Convert rows to list of dictionaries
            rows = []
            for row in result:
                # Convert each value to a serializable format
                row_dict = {}
                for col, val in zip(column_names, row):
                    if isinstance(val, datetime):
                        row_dict[col] = val.isoformat()
                    else:
                        row_dict[col] = val
                rows.append(row_dict)

            execution_time = time.time() - start_time
            
            return {
                "status": "success",
                "execution_time": execution_time,
                "row_count": len(rows),
                "columns": list(column_names),
                "data": rows
            }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "execution_time": time.time() - start_time
        }


@mcp.tool(
    name="Test_Connection",
    description="Test if a database connection is valid"
)
async def test_connection(connection_string: str) -> Dict[str, Any]:
    """Test if a connection string is valid"""
    try:
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {
            "status": "success",
            "message": "Connection successful"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@mcp.tool(
    name="Get_OMOP_Schema",
    description="Get the OMOP CDM schema information for prompting"
)
def get_omop_schema() -> Dict[str, Any]:
    """Load and format OMOP CDM schema for prompting"""
    try:
        config = get_config()
        schema_path = os.path.join(
            config["database"]["schema_directory"],
            config["omop_cdm"]["schema_file"]
        )
        print(f"Loading schema from: {schema_path}")  # Debug print

        with open(schema_path, "r") as f:
            schema_data = json.load(f)

        # Format the schema for the prompt
        schema_text = "OMOP CDM Database Schema:\n\n"

        # Add main tables first
        core_tables = ["person", "visit_occurrence", "condition_occurrence", "drug_exposure", "measurement",
                       "observation"]

        # First add core tables for better context
        schema_text += "Core Tables:\n"
        for table_name in core_tables:
            table = next((t for t in schema_data["tables"] if t["name"] == table_name), None)
            if table:
                schema_text += f"Table: {table['name']}\n"

                for col in table["columns"]:
                    required = col.get("required", False)
                    req_text = " (required)" if required else ""
                    schema_text += f"  - {col['name']} ({col['type']}){req_text}\n"

                schema_text += "\n"

        # Then add other tables
        schema_text += "Other Tables:\n"
        for table in schema_data["tables"]:
            if table["name"] not in core_tables:
                schema_text += f"Table: {table['name']}\n"

                # Add primary columns for non-core tables
                key_columns = [c for c in table["columns"] if
                               c.get("required", False) or "_id" in c["name"] or "concept_id" in c["name"]]
                for col in key_columns:
                    required = col.get("required", False)
                    req_text = " (required)" if required else ""
                    schema_text += f"  - {col['name']} ({col['type']}){req_text}\n"

                schema_text += f"  - plus {len(table['columns']) - len(key_columns)} more columns\n\n"

        # Add relationships
        if "relationships" in schema_data:
            schema_text += "\nKey Relationships:\n"
            for relation in schema_data["relationships"]:
                source_table = relation["source_table"]
                target_table = relation["target_table"]
                join_columns = relation["join_columns"]

                for source_col, target_col in join_columns.items():
                    schema_text += f"- {source_table}.{source_col} -> {target_table}.{target_col}\n"

        # Add common joins if available
        if "common_joins" in schema_data:
            schema_text += "\nCommon Join Patterns:\n"
            for join in schema_data["common_joins"]:
                schema_text += f"- {join['name']}: {join['description']}\n"
                schema_text += f"  {join['sql_pattern']}\n"

        return {
            "status": "success",
            "schema_text": schema_text,
            "raw_schema": schema_data
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


# ============== VALIDATION SERVER TOOLS ===============

@mcp.tool(
    name="Validate_SQL_Query",
    description="Validate a SQL query against OMOP CDM validation rules"
)
def validate_query(sql_query: str) -> Dict[str, Any]:
    """Validate a SQL query against OMOP CDM rules using regex"""
    validation_result = {
        "is_valid": True,
        "issues": []
    }

    # Convert SQL to lowercase for case-insensitive checks
    sql_lower = sql_query.lower()

    # Check for prohibited operations
    prohibited_patterns = [
        r'\bdrop\s+table\b',
        r'\btruncate\s+table\b',
        r'\bdelete\s+from\b',
        r'\bupdate\s+\w+\s+set\b',
        r'\balter\s+table\b'
    ]

    for pattern in prohibited_patterns:
        if re.search(pattern, sql_lower):
            validation_result["is_valid"] = False
            validation_result["issues"].append(
                "Query contains prohibited operations (DROP, TRUNCATE, DELETE, UPDATE, ALTER)")
            break

    # Extract table references - improved to handle aliases
    table_pattern = r'from\s+([a-zA-Z0-9_]+)(?:\s+(?:as\s+)?([a-zA-Z0-9_]+))?|join\s+([a-zA-Z0-9_]+)(?:\s+(?:as\s+)?([a-zA-Z0-9_]+))?'
    table_matches = re.finditer(table_pattern, sql_lower)
    
    tables = {}
    for match in table_matches:
        if match.group(1):  # FROM clause
            table_name = match.group(1)
            alias = match.group(2) or table_name
            tables[alias] = table_name
        elif match.group(3):  # JOIN clause
            table_name = match.group(3)
            alias = match.group(4) or table_name
            tables[alias] = table_name
    
    # Check for required tables
    required_tables = VALIDATION_RULES.get("required_tables", [])
    for table in required_tables:
        trigger = table["when"].lower()
        required_name = table["name"].lower()
        
        # Check if trigger is mentioned but required table is missing
        table_names = list(tables.values())
        if any(trigger in name for name in table_names) and not any(required_name == name for name in table_names):
            validation_result["is_valid"] = False
            validation_result["issues"].append(
                f"Missing required table {table['name']} when querying {table['when']}")

    # Check for required joins with improved alias handling
    required_joins = VALIDATION_RULES.get("required_joins", [])
    for join in required_joins:
        table1 = join["table1"].lower()
        table2 = join["table2"].lower()
        
        # Get aliases for these tables
        table1_aliases = [alias for alias, name in tables.items() if name == table1]
        table2_aliases = [alias for alias, name in tables.items() if name == table2]
        
        # Add original table names to aliases list
        table1_aliases.append(table1)
        table2_aliases.append(table2)
        
        # Check if both tables are in the query
        if any(table1 == name for name in tables.values()) and any(table2 == name for name in tables.values()):
            # Generate possible join conditions
            join_patterns = []
            for t1_alias in table1_aliases:
                for t2_alias in table2_aliases:
                    join_patterns.extend([
                        f"{t1_alias}\\.[a-z0-9_]+\\s*=\\s*{t2_alias}\\.[a-z0-9_]+",
                        f"{t2_alias}\\.[a-z0-9_]+\\s*=\\s*{t1_alias}\\.[a-z0-9_]+"
                    ])
            
            # Check if any join pattern is found
            if not any(re.search(pattern, sql_lower) for pattern in join_patterns):
                validation_result["is_valid"] = False
                validation_result["issues"].append(
                    f"Missing proper join condition between {join['table1']} and {join['table2']}")

    # Check for date range filters on temporal queries
    if any(term in sql_lower for term in ["date", "datetime", "time"]):
        if not any(term in sql_lower for term in ["between", ">", "<", ">=", "<="]):
            validation_result["issues"].append("Warning: Temporal query without date range filter")

    # Check for basic SQL syntax issues (unbalanced parentheses, missing quotes)
    if sql_query.count('(') != sql_query.count(')'):
        validation_result["is_valid"] = False
        validation_result["issues"].append("SQL syntax error: Unbalanced parentheses")

    # Check for unclosed quotes
    quote_chars = ["'", '"']
    for quote in quote_chars:
        if sql_query.count(quote) % 2 != 0:
            validation_result["is_valid"] = False
            validation_result["issues"].append(f"SQL syntax error: Unclosed {quote} quotes")

    return validation_result


# ============== OLLAMA SERVER TOOLS ===============

@mcp.tool(
    name="Generate_SQL",
    description="Generate SQL from natural language using an LLM"
)
async def generate_sql(prompt: str, schema: str, medical_concepts: Dict[str, Any] = None, model_name: str = None, system_prompt: str = None) -> Dict[str, Any]:
    """Generate SQL from natural language using Ollama
    
    Args:
        prompt: Natural language query
        schema: OMOP CDM schema information
        medical_concepts: Optional dictionary containing extracted medical concepts with their codes
        model_name: Optional model name override
        system_prompt: Optional system prompt override
    """
    config = get_config()

    # Set up the model name and API URL
    model = model_name or config["ollama"]["default_model"]
    api_url = config["ollama"]["api_url"]  # Use the URL from config

    # Default system prompt for SQL generation
    default_system = "You are an expert in SQL and healthcare data analysis, specifically working with the OMOP Common Data Model (CDM)."
    system_message = system_prompt or default_system

    # Build the base prompt
    full_prompt = f"""System: {system_message}

Given the following OMOP CDM schema:

{schema}

IMPORTANT: You are generating SQL for the OMOP Common Data Model (CDM). Your task is to convert natural language queries into precise SQL that follows OMOP CDM best practices.

KEY REQUIREMENTS:
1. For patient age queries:
   - Use person.year_of_birth and CURRENT_DATE to calculate age
   - Example: EXTRACT(YEAR FROM CURRENT_DATE) - person.year_of_birth > 65
2. For patient counts:
   - Always use COUNT(DISTINCT person.person_id)
   - Always join with the person table
3. For clinical conditions:
   - Join with concept table using appropriate _concept_id
   - Use concept.concept_name for text matching
   - Use concept.vocabulary_id for standard vocabularies

Convert the following natural language query into a valid SQL query:

"{prompt}"

Your response must:
1. Contain ONLY the SQL query
2. No explanations or markdown formatting
3. No ```sql or ``` markers
4. End with a semicolon

Example format:
SELECT COUNT(DISTINCT person.person_id) as patient_count 
FROM person 
WHERE EXTRACT(YEAR FROM CURRENT_DATE) - person.year_of_birth > 65;

Your SQL query (and nothing else):
"""

    # Add medical concepts if provided
    if medical_concepts:
        concepts_text = "\nExtracted Medical Concepts:\n"
        for category, concepts in medical_concepts.items():
            if concepts:  # Only add categories that have concepts
                concepts_text += f"\n{category.upper()}:\n"
                for concept in concepts:
                    concepts_text += f"- {concept['concept_name']} (ID: {concept['concept_id']}, Vocabulary: {concept['vocabulary_id']})\n"
        full_prompt += concepts_text

    print(f"Using model: {model}")
    print(f"Full prompt: {full_prompt}")

    # Prepare the request for Ollama
    ollama_request = {
        "model": model,
        "prompt": full_prompt,
        "stream": False  # Disable streaming for now to simplify the response handling
    }

    try:
        # Use httpx for async HTTP requests
        async with httpx.AsyncClient() as client:
            print(f"Sending request to {api_url}")
            response = await client.post(api_url, json=ollama_request, timeout=240)
            print(f"Response status: {response.status_code}")
            response.raise_for_status()

            # Get the response data
            response_data = response.json()
            print(f"Response data: {response_data}")

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
            else:
                error_msg = "No response field in Ollama API response"
                print(error_msg)
                print(f"Response was: {response_data}")
                return {
                    "status": "error",
                    "error": error_msg
                }

            # Verify we have a valid SQL query
            if not sql_query or not sql_query.strip().upper().startswith("SELECT"):
                error_msg = "Generated response does not contain a valid SQL query"
                print(error_msg)
                print(f"Response was: {sql_query}")
                return {
                    "status": "error",
                    "error": error_msg
                }

            print(f"Final SQL query: {sql_query}")
            return {
                "status": "success",
                "sql_query": sql_query,
                "confidence": 0.9
            }

    except httpx.HTTPError as e:
        error_msg = f"HTTP error occurred: {str(e)}"
        print(error_msg)
        return {
            "status": "error",
            "error": error_msg
        }
    except Exception as e:
        error_msg = f"Error generating SQL: {str(e)}"
        print(error_msg)
        return {
            "status": "error",
            "error": error_msg
        }


@mcp.tool(
    name="Generate_Explanation",
    description="Generate an explanation for an SQL query"
)
async def generate_explanation(sql_query: str, model_name: str = None) -> Dict[str, Any]:
    """Generate an explanation for the SQL query"""
    config = get_config()
    model = model_name or config["ollama"]["default_model"]
    api_url = "http://localhost:11434"  # Base URL without /api/generate

    explanation_prompt = f"""System: You are a healthcare analytics expert explaining SQL queries to clinicians.

Explain what this SQL query does in simple terms, focusing on the healthcare insights it provides:

```sql
{sql_query}
```

Explain in 2-3 sentences what clinical question this query answers and how it uses the OMOP CDM structure.
"""

    explanation_request = {
        "model": model,
        "prompt": explanation_prompt,
        "stream": True  # Enable streaming to get chunks
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{api_url}/api/generate", json=explanation_request, timeout=120)
            response.raise_for_status()
            
            # Process the streaming response
            full_response = ""
            async for line in response.aiter_lines():
                try:
                    chunk = json.loads(line)
                    if "response" in chunk:
                        full_response += chunk["response"]
                except json.JSONDecodeError:
                    continue
            
            return {
                "status": "success",
                "explanation": full_response.strip()
            }
    except httpx.HTTPError as e:
        error_msg = f"HTTP error occurred: {str(e)}"
        print(error_msg)
        return {
            "status": "error",
            "error": error_msg
        }
    except Exception as e:
        error_msg = f"Error generating explanation: {str(e)}"
        print(error_msg)
        return {
            "status": "error",
            "error": error_msg
        }


@mcp.tool(
    name="Generate_Answer",
    description="Generate a natural language answer based on query, SQL, and results"
)
async def generate_answer(question: str, sql_query: str, results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a natural language answer to the original question"""
    config = get_config()
    api_url = "http://localhost:11434"  # Base URL without /api/generate
    model = config["ollama"]["default_model"]

    # Convert results to a readable format
    results_text = json.dumps(results, indent=2)

    answer_prompt = f"""System: You are a healthcare analytics expert explaining query results to clinicians.

Given the following:

Question: "{question}"

SQL Query:
```sql
{sql_query}
```

Query Results:
```
{results_text}
```

Generate a comprehensive natural language answer to the original question based on these results.
Explain the insights from the data in a way that would be understandable to healthcare professionals.
"""

    answer_request = {
        "model": model,
        "prompt": answer_prompt,
        "stream": False
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{api_url}/api/generate", json=answer_request, timeout=240)
            response.raise_for_status()
            return {
                "status": "success",
                "answer": response.json()["response"].strip()
            }
    except httpx.HTTPError as e:
        error_msg = f"HTTP error occurred: {str(e)}"
        print(error_msg)
        return {
            "status": "error",
            "error": error_msg
        }
    except Exception as e:
        error_msg = f"Error generating answer: {str(e)}"
        print(error_msg)
        return {
            "status": "error",
            "error": error_msg
        }


@mcp.tool(
    name="List_Available_Models",
    description="List available LLM models from Ollama"
)
async def list_available_models() -> Dict[str, Any]:
    """List available models from Ollama"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{api_url}/tags", timeout=10)
            response.raise_for_status()
            return {
                "status": "success",
                "models": response.json().get("models", [])
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


# Add health check endpoint
@mcp.tool(
    name="Health_Check",
    description="Check if the MCP server is running"
)
def health_check() -> Dict[str, Any]:
    """Check if the MCP server is running"""
    return {
        "status": "success",
        "message": "MCP server is running"
    }


# Start the server when run directly
if __name__ == "__main__":
    import socket
    import time
    
    def is_port_available(port: int) -> bool:
        """Check if a port is available"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                return True
            except OSError:
                return False
    
    # Try to start the server with retries
    max_retries = 5
    retry_count = 0
    port = 8000
    
    while retry_count < max_retries:
        if is_port_available(port):
            print(f"Starting MCP server on port {port}")
            try:
                mcp.run(host="0.0.0.0", port=port)
                break
            except Exception as e:
                print(f"Error starting server: {e}")
                retry_count += 1
                time.sleep(2)
        else:
            print(f"Port {port} is in use, trying again in 2 seconds...")
            retry_count += 1
            time.sleep(2)
    
    if retry_count >= max_retries:
        print("Failed to start MCP server after multiple attempts")
        import sys
        sys.exit(1)