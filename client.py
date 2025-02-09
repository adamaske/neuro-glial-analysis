
import socket
from pylsl import StreamInfo, StreamOutlet

def tcp_client(host='192.168.0.100', port=32000):
    try:

        # Create a socket object
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Connect to the server
        client_socket.connect((host, port))
        print(f'Connected to {host}:{port}')
        
        # Send data to the server
        message = "Hello, from Client!"
        client_socket.sendall(message.encode('utf-8'))


        print(f'Sent: {message}')
        
        # Receive data from the server
        response = client_socket.recv(1024)
        print(f'Received: {response.decode("utf-8")}')
        
    except Exception as e:
        print(f'Error: {e}')
    finally:
        # Close the connection
        client_socket.close()
        print('Connection closed')

if __name__ == "__main__":
    tcp_client()
