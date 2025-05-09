import asyncio
import subprocess
import time
import sys
import os
import socket
import psutil
from agent2agent_omcp.data_agent import EhrRetrievalAgent
from python_a2a import run_server
import dotenv

def is_port_in_use(port: int) -> bool:
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def kill_process_on_port(port: int):
    """Kill any process using the specified port"""
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            # Get all connections for the process
            connections = proc.connections()
            for conn in connections:
                if hasattr(conn, 'laddr') and conn.laddr.port == port:
                    print(f"Killing process {proc.pid} ({proc.name()}) using port {port}")
                    proc.kill()
                    time.sleep(1)  # Give it time to release the port
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

async def start_mcp_server():
    """Start the MCP server"""
    print("Starting MCP server...")
    
    # Check if port 8000 is in use
    if is_port_in_use(8000):
        print("Port 8000 is in use. Attempting to free it...")
        kill_process_on_port(8000)
        time.sleep(2)  # Wait for port to be freed
    
    # Get the path to data_mcp.py
    mcp_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_mcp.py")
    
    # Start the MCP server as a subprocess
    mcp_process = subprocess.Popen(
        [sys.executable, mcp_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True  # Use text mode for easier output handling
    )
    
    # Wait for the server to start and check its output
    max_retries = 5
    retry_count = 0
    while retry_count < max_retries:
        # Check if process is still running
        if mcp_process.poll() is not None:
            stdout, stderr = mcp_process.communicate()
            print("Error starting MCP server:")
            print(stderr)
            sys.exit(1)
        
        # Try to connect to the server
        if not is_port_in_use(8000):
            print("Waiting for MCP server to start...")
            time.sleep(2)
            retry_count += 1
            continue
            
        # Check server health
        try:
            import httpx
            response = httpx.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                print("MCP server started successfully")
                return mcp_process
        except Exception as e:
            print(f"Waiting for MCP server to be ready... ({retry_count + 1}/{max_retries})")
            time.sleep(2)
            retry_count += 1
            continue
    
    print("Failed to start MCP server after multiple attempts")
    mcp_process.terminate()
    sys.exit(1)

async def start_agent_server():
    """Start the agent server"""
    print("Starting agent server...")
    
    # Check if port 8001 is in use
    if is_port_in_use(8001):
        print("Port 8001 is in use. Attempting to free it...")
        kill_process_on_port(8001)
        time.sleep(2)  # Wait for port to be freed
    
    # Create and start the agent server
    agent = EhrRetrievalAgent()
    
    # Start the server and wait for it to be ready
    run_server(agent, host="localhost", port=8001)

async def main():
    try:
        # Start MCP server first
        mcp_process = await start_mcp_server()
        
        # Then start agent server
        agent_task = await start_agent_server()
        
        print("\nBoth servers are running!")
        print("MCP server: http://localhost:8000")
        print("Agent server: http://localhost:8001")
        print("\nPress Ctrl+C to stop both servers...")
        
        # Keep the script running until interrupted
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        # Terminate MCP process
        mcp_process.terminate()
        # Cancel agent server task
        agent_task.cancel()
        print("Servers stopped")

if __name__ == "__main__":
    asyncio.run(main()) 