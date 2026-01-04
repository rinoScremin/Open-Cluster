#include "ggml.h"
#include "ggml-backend.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdlib>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <csignal>
#include <zmq.hpp>
#include <filesystem>
#include <set>
#include <array>
#include <cstdio>
#include <sstream>
#include <map>
#include "matrix_backend.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cmath>
#include <list>
#include <cstdint>
#include <algorithm>
#include <torch/torch.h>

struct combined_matrix_shards
{
    int total_shards_reserved = 0;        // Number of shards currently received
    int number_of_shards_needed = 0;      // Total shards expected for this matrix
    std::string file_name;                // Base filename (without shard index)
    
    std::vector<int> shard_numbers;       // List of received shard indices
    std::list<std::vector<uint8_t>> received_matrix_data;  // Raw binary data of each shard
    std::list<std::vector<int>> dims_list;                 // Dimensions of each shard [batch, depth, rows, cols]
    

    int join_dim = 0; // << for now you only will join dim=0 but join based off this 
    // Note: Using std::list for received data allows efficient insertion
    //       as shards arrive in potentially non-sequential order from workers

    std::vector<int> hierarchical_split_order;


};

// Function to execute a shell command and capture its output
std::string exec_command(const char* cmd)
{
    // Buffer to store chunks of command output
    std::array<char, 128> buffer;
    // String to accumulate the full command output
    std::string result;
    
    // Lambda function to safely close the pipe
    // Acts as a custom deleter for the unique_ptr
    auto pipe_closer = [](FILE* pipe) 
    {
        if (pipe) pclose(pipe);
    };
    
    // Create a unique_ptr with custom deleter to ensure pipe cleanup
    // popen() opens a process by creating a pipe and forking/executing the command
    std::unique_ptr<FILE, decltype(pipe_closer)> pipe(popen(cmd, "r"), pipe_closer);
    
    // Check if pipe was successfully created
    if (!pipe) 
    {
        throw std::runtime_error("popen() failed!");
    }
    
    // Read command output chunk by chunk until EOF
    // fgets reads up to buffer.size()-1 characters or until newline/EOF
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) 
    {
        // Append each chunk to the result string
        result += buffer.data();
    }
    
    // Return the complete command output
    return result;
}

// Function to get the local IP address by executing a shell script
std::string get_local_ip() 
{
    try 
    {
        // Execute the shell script that retrieves the LAN interface IP address
        // The script is expected to return the IP address as a string
        std::string ip = exec_command("./get_land_interface.sh");
        
        // Remove trailing newline character if present
        // Shell commands typically output with a newline at the end
        if (!ip.empty() && ip[ip.length()-1] == '\n') 
        {
            ip.erase(ip.length()-1);
        }
        
        // Return the cleaned IP address string
        return ip;
    } 
    catch (const std::exception& e) 
    {
        // Log error if command execution fails
        std::cerr << "Error getting local IP: " << e.what() << std::endl;
        
        // Return localhost address as fallback in case of failure
        return "127.0.0.1";
    }
}

std::string get_env(const char* env_var, const char* default_val) 
{
    const char* env_value = std::getenv(env_var);
    return env_value ? std::string(env_value) : std::string(default_val);
}

class llama_zmq_server 
{       
    public:   
        std::string project_folder;
        std::string matrix_shard_folder;
        std::string matrix_results_folder;
        std::string head_node_ip_eth;
        std::string head_node_ip_wifi;
        std::string head_node_PULL_port;
        std::string head_node_PUSH_port;
        std::string worker_node_PULL_port;
        std::string worker_node_PUSH_port;
        
        std::string local_IP_eth;
        std::string local_IP_wifi;
        
        std::string eth_pull_port;
        std::string eth_push_port;
        std::string wifi_pull_port;
        std::string wifi_push_port;
        std::string worker_peer_port;
        
        zmq::context_t zmq_context;
        zmq::socket_t file_receiver_eth;
        zmq::socket_t file_sender_eth;
        zmq::socket_t file_receiver_wifi;
        zmq::socket_t file_sender_wifi;
        zmq::socket_t head_node_sender_eth;
        zmq::socket_t head_node_sender_wifi;
        zmq::socket_t ack_sender;  // For sending ACKs to Python front-end
        zmq::socket_t worker_peer_receiver; // Worker ↔ Worker peer communication
        // Add this to your class member variables  
        zmq::socket_t ack_receiver;  // For receiving ACK confirmations from Python  
        
        // Unified reserved file structure to hold incoming files from any interface
        struct ReservedFiles {
            std::vector<std::string> save_parallel_file_name; // Filename(s) for parallel or single-file saves (use [0] for single)
            std::vector<uint8_t> received_data_eth_file;      // Data received via Ethernet interface
            std::vector<uint8_t> received_data_wifi_file;     // Data received via WiFi interface
            bool is_parallel = false;                         // True when this ReservedFiles holds ETH+WiFi halves
            bool processed = false;                           // Marked once processed by save_file_handler
        };

        // Central list that holds all incoming files (Ethernet, WiFi, parallel)
        std::vector<ReservedFiles> reserved_files_list;

        std::vector<combined_matrix_shards> combined_matrix_shards_list;

        // In your class member variables:
        std::vector<std::string> matrix_file_paths;

        std::vector<std::string> received_data_eth_linux_command;
        std::vector<std::string> received_data_wifi_linux_command;
        std::vector<std::string> received_data_eth_server_command;
        std::vector<std::string> received_data_wifi_server_command;
        
        // Thread-safe mutexes (ADD THESE)
        std::mutex linux_commands_mutex;
        std::mutex server_commands_mutex;
        std::mutex file_data_mutex;
        std::mutex wifi_commands_mutex;
        
        std::atomic<bool> server_running;
        llama_matrix_backend matrix_backend_llama;

        int send_back_number_of_shards = 0;
        std::vector<std::string> worker_ip_list;
        std::vector<float> worker_percentages;


        std::map<std::string, std::vector<std::pair<int, std::vector<uint8_t>>>> pending_shards;  
        std::map<std::string, std::set<int>> received_shards;  
        std::mutex shared_memory_mutex;
        // Fallback shard counters for outputs when inputs have no shard suffix
        std::map<std::string, int> output_shard_counters;
        std::mutex output_shard_mutex;


    public:            
        // Constructor - initializes ZMQ server with dual network interfaces
        llama_zmq_server() : 
            zmq_context(1),
            file_receiver_eth(zmq_context, zmq::socket_type::pull),
            file_sender_eth(zmq_context, zmq::socket_type::push),
            file_receiver_wifi(zmq_context, zmq::socket_type::pull),
            file_sender_wifi(zmq_context, zmq::socket_type::push),
            head_node_sender_eth(zmq_context, zmq::socket_type::push),
            head_node_sender_wifi(zmq_context, zmq::socket_type::push),
            worker_peer_receiver(zmq_context, zmq::socket_type::pull),
            server_running(true)
        {
            // Load configuration from environment variables with defaults
            project_folder = get_env("OPEN_CLUSTER_PROJECT_DIRECTORY", 
                                "/home/rino/Desktop/Open_Cluster_AI_Station_beta/");
            matrix_shard_folder = get_env("OPEN_CLUSTER_MATRIX_SHARD_DIRECTORY", 
                                        "/dev/shm/matrix_shards/");
            matrix_results_folder = get_env("OPEN_CLUSTER_MATRIX_RESULTS_DIRECTORY", 
                                        "/dev/shm/matrix_results/");
            
            head_node_ip_eth = get_env("HEAD_NODE_IP_ETH", "192.168.2.100");
            head_node_ip_wifi = get_env("HEAD_NODE_IP_WIFI", "192.168.50.113");
            head_node_PULL_port = get_env("HEAD_NODE_PULL_PORT_C", "7779");
            head_node_PUSH_port = get_env("HEAD_NODE_PUSH_PORT_C", "7780");
            worker_node_PULL_port = get_env("WORKER_NODE_PULL_PORT_C", "5557");
            worker_node_PUSH_port = get_env("WORKER_NODE_PUSH_PORT_C", "5558");
            
            // Initialize parallel file structures (now handled via reserved_files_list)
            
            // Get local network addresses
            local_IP_eth = get_local_ip();
            
            // Attempt to get WiFi IP address using system command
            try {
                local_IP_wifi = exec_command(
                    "ip -4 addr show $(ip -4 route ls | grep default | grep -o 'dev [^ ]*' "
                    "| awk '{print $2}') | grep inet | awk '{print $2}' | cut -d'/' -f1"
                );
                // Clean up newline from command output
                if (!local_IP_wifi.empty() && local_IP_wifi[local_IP_wifi.length()-1] == '\n') {
                    local_IP_wifi.erase(local_IP_wifi.length()-1);
                }
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to get WiFi IP: " << e.what() << std::endl;
                local_IP_wifi = "127.0.0.1";
            }
            
            // Configure network ports based on whether this is head node or worker node
            if (local_IP_eth == head_node_ip_eth || local_IP_wifi == head_node_ip_wifi) {
                // Head node configuration
                eth_pull_port = "tcp://" + local_IP_eth + ":" + head_node_PULL_port;
                eth_push_port = "tcp://" + local_IP_eth + ":" + head_node_PUSH_port;
                wifi_pull_port = "tcp://" + local_IP_wifi + ":" + head_node_PULL_port;
                wifi_push_port = "tcp://" + local_IP_wifi + ":" + head_node_PUSH_port;
            } else {
                // Worker node configuration
                eth_pull_port = "tcp://" + local_IP_eth + ":" + worker_node_PULL_port;
                eth_push_port = "tcp://" + local_IP_eth + ":" + worker_node_PUSH_port;
                wifi_pull_port = "tcp://" + local_IP_wifi + ":" + worker_node_PULL_port;
                wifi_push_port = "tcp://" + local_IP_wifi + ":" + worker_node_PUSH_port;
            }
            
            
            // In constructor (after ack_sender setup)  
            ack_receiver = zmq::socket_t(zmq_context, zmq::socket_type::pull);  
            ack_receiver.bind("tcp://0.0.0.0:7791");  // Different port than ack_sender

            // Bind file transfer sockets
            file_receiver_eth.bind(eth_pull_port);
            file_sender_eth.bind(eth_push_port);
            file_receiver_wifi.bind(wifi_pull_port);
            file_sender_wifi.bind(wifi_push_port);
            
            // Connect to head node for coordination
            head_node_sender_eth.connect("tcp://" + head_node_ip_eth + ":" + head_node_PULL_port);
            head_node_sender_wifi.connect("tcp://" + head_node_ip_wifi + ":" + head_node_PULL_port);
            
            // Setup Python front-end ACK communication
            std::string python_frontend_ip = get_env("HEAD_NODE_IP", "192.168.2.100");
            std::string python_frontend_port = get_env("PYTHON_FRONT_END_CLUSTER_PORT", "7790");
            
            ack_sender = zmq::socket_t(zmq_context, zmq::socket_type::push);
            ack_sender.connect("tcp://" + python_frontend_ip + ":" + python_frontend_port);
            
            // Experimental feature - work distribution percentages for heterogeneous nodes
            // This enables adaptive load balancing based on worker capabilities
            worker_percentages = {0.45f, 0.35f, 0.10f, 0.05f, 0.05f};  // For experimental feature not yet implemented
            
            worker_peer_port = get_env("WORKER_PEER_PORT", "5560");
            worker_peer_receiver.bind("tcp://" + local_IP_eth + ":" + worker_peer_port);
            worker_peer_receiver.bind("tcp://" + local_IP_wifi + ":" + worker_peer_port);
            
            // Clean console output
            std::cout << "\n=== ZMQ Server Initialization ===" << std::endl;
            std::cout << "Network Configuration:" << std::endl;
            std::cout << "  Ethernet IP: " << local_IP_eth << std::endl;
            std::cout << "  WiFi IP: " << local_IP_wifi << std::endl;
            std::cout << "\nPort Bindings:" << std::endl;
            std::cout << "  Ethernet PULL: " << eth_pull_port << std::endl;
            std::cout << "  Ethernet PUSH: " << eth_push_port << std::endl;
            std::cout << "  WiFi PULL: " << wifi_pull_port << std::endl;
            std::cout << "  WiFi PUSH: " << wifi_push_port << std::endl;
            std::cout << "  Worker Peer: " << worker_peer_port << std::endl;
            std::cout << "  Worker IPs configured: " << worker_ip_list.size() << " nodes" << std::endl;
            
            // Initialize hardware backends
            #ifdef GGML_OPENCL
                std::cout << "\nInitializing OpenCL backends..." << std::endl;
                init_openCL_GPUS();
            #else
                std::cout << "\nOpenCL backend disabled at compile time" << std::endl;
            #endif
            
            std::cout << "\nServer initialization complete" << std::endl;
            std::cout << "==============================\n" << std::endl;
        }

        void send_ack(std::string ack_msg = "ACK") 
        {
            zmq::message_t ack(ack_msg.data(), ack_msg.size());
            ack_sender.send(ack, zmq::send_flags::none);
        }

        int wait_for_acks(int expected_count, const std::string& expected_msg = "ACK", int timeout_ms = 30000)   
        {  
            int acks = 0;  
            auto start_time = std::chrono::steady_clock::now();  
            
            while (acks < expected_count) {  
                try {  
                    zmq::message_t msg;  
                    auto result = ack_receiver.recv(msg, zmq::recv_flags::dontwait);  
                    
                    if (result) {  
                        std::string received_msg(static_cast<char*>(msg.data()), msg.size());  
                        if (received_msg == expected_msg) {  
                            acks++;  
                            std::cout << "✅ Received " << expected_msg << " " << acks << "/" << expected_count << std::endl;  
                        }  
                    }  
                } catch (const zmq::error_t& e) {  
                    if (e.num() != EAGAIN) {  
                        std::cerr << "❌ ZMQ error receiving ACK: " << e.what() << std::endl;  
                        break;  
                    }  
                }  
                
                // Check timeout  
                auto elapsed = std::chrono::steady_clock::now() - start_time;  
                if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() > timeout_ms) {  
                    std::cout << "⚠️ Timeout waiting for " << expected_msg << " after " << timeout_ms << "ms" << std::endl;  
                    std::cout << "   Received " << acks << "/" << expected_count << " messages" << std::endl;  
                    break;  
                }  
                
                // Brief sleep to avoid 100% CPU  
                std::this_thread::sleep_for(std::chrono::milliseconds(10));  
            }  
            
            if (acks == expected_count) {  
                std::cout << "✅ All ACKs received!" << std::endl;  
            }  
            
            return acks;  
        }

        void run_server() 
        {
            std::cout << "🚀 C++ ZMQ Node Server starting..." << std::endl;
            
            // Start network listener threads for dual-interface operation
            std::thread eth_thread(&llama_zmq_server::listen_interface, this, "Ethernet");
            std::thread wifi_thread(&llama_zmq_server::listen_interface, this, "WiFi");
            std::thread process_command_thread(&llama_zmq_server::process_command, this);
            
            // Detach threads to run as daemon processes (background services)
            eth_thread.detach();
            wifi_thread.detach();
            process_command_thread.detach();
            
            std::cout << "✅ Network listeners started successfully" << std::endl;
            std::cout << "   • Ethernet interface: Active" << std::endl;
            std::cout << "   • WiFi interface: Active" << std::endl;
            std::cout << "   • Command processor: Active" << std::endl;
            std::cout << "\n📡 Server running. Press Ctrl+C to gracefully shutdown..." << std::endl;
            
            try 
            {
                // Register signal handler for graceful shutdown on Ctrl+C
                std::signal(SIGINT, [](int sig) { 
                    std::cout << "\n🛑 Received shutdown signal (Ctrl+C)" << std::endl;
                    std::cout << "   Shutting down ZMQ server..." << std::endl;
                    std::exit(0); 
                });
                
                // Main thread idle loop - keeps the server alive
                // This allows signal handling and keeps the process running
                while (server_running) 
                {
                    // Sleep to prevent CPU spinning while waiting for shutdown
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }
            } 
            catch (const std::exception& e) 
            {
                std::cerr << "\n❌ Critical server error: " << e.what() << std::endl;
                std::cerr << "   Server shutting down due to exception" << std::endl;
            }
            
            std::cout << "👋 Server shutdown complete" << std::endl;
        }

        void listen_interface(const std::string& interface_name)
        {
            // Determine which socket and which command containers/mutex to use
            zmq::socket_t* socket_ptr = nullptr;
            std::vector<std::string>* linux_cmd_ptr = nullptr;
            std::vector<std::string>* server_cmd_ptr = nullptr;
            std::mutex* linux_cmd_mutex = nullptr;
            std::mutex* server_cmd_mutex = nullptr;

            if (interface_name == "Ethernet")
            {
                socket_ptr = &file_receiver_eth;
                linux_cmd_ptr = &received_data_eth_linux_command;
                server_cmd_ptr = &received_data_eth_server_command;
                linux_cmd_mutex = &linux_commands_mutex;
                server_cmd_mutex = &server_commands_mutex;
            }
            else if (interface_name == "WiFi")
            {
                socket_ptr = &file_receiver_wifi;
                linux_cmd_ptr = &received_data_wifi_linux_command;
                server_cmd_ptr = &received_data_wifi_server_command;
                linux_cmd_mutex = &wifi_commands_mutex;
                server_cmd_mutex = &server_commands_mutex;
            }
            else
            {
                std::cerr << "❌ Unknown interface: " << interface_name << std::endl;
                return;
            }

            std::cout << "🔌 " << interface_name << " listener thread started" << std::endl;

            while (server_running)
            {
                try
                {
                    std::vector<zmq::message_t> parts;
                    bool more_parts = true;

                    // Receive multipart ZMQ message (could be 1 or 2 parts)
                    while (more_parts && server_running)
                    {
                        zmq::message_t message;
                        auto result = socket_ptr->recv(message, zmq::recv_flags::dontwait);

                        if (result)
                        {
                            more_parts = message.more();
                            parts.push_back(std::move(message));
                        }
                        else
                        {
                            std::this_thread::sleep_for(std::chrono::milliseconds(10));
                            break;
                        }
                    }

                    if (parts.empty())
                        continue;

                    // Single-part: commands
                    if (parts.size() == 1)
                    {
                        std::string command = parts[0].to_string();
                        size_t server_cmd_pos = command.find("server_command=");

                        if (server_cmd_pos != std::string::npos)
                        {
                            std::string server_cmd = command.substr(server_cmd_pos + 15);
                            std::lock_guard<std::mutex> lock(*server_cmd_mutex);
                            server_cmd_ptr->push_back(server_cmd);
                            std::cout << "📋 " << interface_name << ": Received server command" << std::endl;
                        }
                        else
                        {
                            std::lock_guard<std::mutex> lock(*linux_cmd_mutex);
                            linux_cmd_ptr->push_back(command);
                            std::cout << "💻 " << interface_name << ": Received Linux command" << std::endl;
                        }
                    }
                    // Two-part: file transfer (either full file or parallel half)
                    else if (parts.size() == 2)
                    {
                        std::string filename_header = parts[0].to_string();
                        size_t parallel_send_pos = filename_header.find("P_SEND_");

                        const uint8_t* data_ptr = static_cast<const uint8_t*>(parts[1].data());
                        size_t data_size = parts[1].size();

                        if (parallel_send_pos != std::string::npos)
                        {
                            // Parallel half (ETH or WiFi)
                            std::string actual_filename = filename_header.substr(parallel_send_pos + 7);

                            std::lock_guard<std::mutex> lock(file_data_mutex);

                            bool found = false;
                            for (auto& rf : reserved_files_list)
                            {
                                if (!rf.save_parallel_file_name.empty() &&
                                    rf.save_parallel_file_name[0] == actual_filename)
                                {
                                    if (interface_name == "Ethernet")
                                        rf.received_data_eth_file.assign(data_ptr, data_ptr + data_size);
                                    else
                                        rf.received_data_wifi_file.assign(data_ptr, data_ptr + data_size);

                                    rf.is_parallel = true;
                                    found = true;
                                    std::cout << "📂 " << interface_name
                                            << ": Added to parallel file '"
                                            << actual_filename << "'" << std::endl;
                                    break;
                                }
                            }

                            if (!found)
                            {
                                ReservedFiles rf;
                                rf.save_parallel_file_name.push_back(actual_filename);

                                if (interface_name == "Ethernet")
                                    rf.received_data_eth_file.assign(data_ptr, data_ptr + data_size);
                                else
                                    rf.received_data_wifi_file.assign(data_ptr, data_ptr + data_size);

                                rf.is_parallel = true;
                                reserved_files_list.push_back(std::move(rf));

                                std::cout << "📂 " << interface_name
                                        << ": Started parallel file '"
                                        << actual_filename << "' ("
                                        << interface_name << " half)" << std::endl;
                            }
                        }
                        else
                        {
                            // Full file transfer over single interface
                            std::string filename = filename_header;

                            std::vector<uint8_t> file_data;
                            file_data.assign(data_ptr, data_ptr + data_size);

                            {
                                std::lock_guard<std::mutex> lock(file_data_mutex);
                                bool found = false;

                                for (auto& rf : reserved_files_list)
                                {
                                    if (!rf.save_parallel_file_name.empty() &&
                                        rf.save_parallel_file_name[0] == filename)
                                    {
                                        if (interface_name == "Ethernet")
                                            rf.received_data_eth_file = std::move(file_data);
                                        else
                                            rf.received_data_wifi_file = std::move(file_data);

                                        found = true;
                                        break;
                                    }
                                }

                                if (!found)
                                {
                                    ReservedFiles rf;
                                    rf.save_parallel_file_name.push_back(filename);

                                    if (interface_name == "Ethernet")
                                        rf.received_data_eth_file = std::move(file_data);
                                    else
                                        rf.received_data_wifi_file = std::move(file_data);

                                    reserved_files_list.push_back(std::move(rf));
                                }
                            }

                            std::cout << "📁 " << interface_name
                                    << ": Received file '" << filename
                                    << "' (" << data_size << " bytes)" << std::endl;
                        }

                        // Attempt to process saved files
                        save_file_handler();
                    }
                    else
                    {
                        std::cout << "⚠️ " << interface_name
                                << ": Unexpected message format - "
                                << parts.size() << " parts received" << std::endl;
                    }
                }
                catch (const std::exception& e)
                {
                    std::cerr << "❌ " << interface_name
                            << " listener error: " << e.what() << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }

            std::cout << "🔌 " << interface_name << " listener thread stopping" << std::endl;
        }

        void process_command() 
        {
            std::cout << "⚙️ Command processor thread started" << std::endl;
            
            while (server_running) 
            {
                try 
                {
                    // --- Process Linux System Commands (Ethernet) ---
                    if (!received_data_eth_linux_command.empty()) 
                    {
                        std::vector<std::string> commands_to_process;
                        
                        // Safely copy commands from shared vector under lock
                        {
                            std::lock_guard<std::mutex> lock(linux_commands_mutex);
                            commands_to_process = received_data_eth_linux_command;
                            received_data_eth_linux_command.clear();  // Clear after copying
                        }
                        
                        std::cout << "\n🔧 Processing " << commands_to_process.size() 
                                << " Linux command(s)" << std::endl;
                        
                        for (const std::string &command : commands_to_process)
                        {
                            // Security note: system() calls should be validated in production
                            std::cout << "   • Executing: " << command << std::endl;
                            
                            int result = system(command.c_str());
                            if (result == 0) {
                                std::cout << "     ✅ Command completed successfully" << std::endl;
                            } else {
                                std::cout << "     ⚠️ Command returned exit code: " << result << std::endl;
                            }
                        }
                    }
                    
                    // --- Process Server Control Commands (Ethernet) ---
                    if (!received_data_eth_server_command.empty()) 
                    {
                        std::vector<std::string> server_commands_to_process;
                        
                        // Safely copy server commands from shared vector
                        {
                            std::lock_guard<std::mutex> lock(server_commands_mutex);
                            server_commands_to_process = received_data_eth_server_command;
                            received_data_eth_server_command.clear();  // Clear after copying
                        }
                        
                        std::cout << "\n🎮 Processing " << server_commands_to_process.size() 
                                << " server control command(s)" << std::endl;
                        
                        // Create threads for concurrent server command execution
                        std::vector<std::thread> command_threads;
                        
                        for (const std::string &command : server_commands_to_process)
                        {
                            std::cout << "   • Launching command: " 
                                    << (command.length() > 50 ? command.substr(0, 47) + "..." : command) 
                                    << std::endl;
                            
                            // Launch each server command in its own thread for parallel execution
                            command_threads.emplace_back([this, command]() {
                                try {
                                    run_server_command(command);
                                } catch (const std::exception& e) {
                                    std::cerr << "❌ Server command failed: " << e.what() 
                                            << " (Command: " << command << ")" << std::endl;
                                }
                            });
                        }
                        
                        // Detach threads to allow them to run independently
                        // Note: Using detach() means we don't wait for completion
                        // Use join() if synchronization is required
                        for (auto& thread : command_threads) {
                            thread.detach();
                        }
                        
                        std::cout << "     ✅ " << command_threads.size() 
                                << " command thread(s) launched" << std::endl;
                    }
                    
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    
                } 
                catch (const std::exception& e) 
                {
                    std::cerr << "❌ Command processor thread error: " << e.what() << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }
            
            std::cout << "⚙️ Command processor thread stopping" << std::endl;
        }

        int run_server_command(const std::string& command) 
        {
            try 
            {
                std::cout << "🚀 Executing server command" << std::endl;
                
                // Tokenize command string into individual arguments
                std::vector<std::string> command_args;
                std::istringstream iss(command);
                std::string token;
                
                while (iss >> token) {
                    command_args.push_back(token);
                }

                // Validate minimum command structure
                if (command_args.empty()) {
                    std::cerr << "❌ Empty command received" << std::endl;
                    return -2;
                }

                const std::string& command_type = command_args[0];

                // ----------------------------
                // Matrix Computation Operations
                // ----------------------------
                if (command_type == "llama" || command_type == "opencl" || command_type == "torch") 
                {
                    // Validate required parameters for matrix operations
                    if (command_args.size() < 10) {
                        std::cerr << "❌ Insufficient parameters for " << command_type 
                                << " operation (expected 10, got " << command_args.size() << ")" << std::endl;
                        return -3;
                    }

                    // Parse matrix operation parameters
                    bool transposeA = (command_args[2] == "true");
                    bool transposeB = (command_args[4] == "true");
                    bool use_gpu    = (command_args[5] == "true");
                    int gpu_id      = std::stoi(command_args[6]);
                    std::string send_back_str = command_args[7];  // Get as string for parsing
                    std::string operation_type = command_args[8];  // e.g., "matmul", "add", etc.
                    int n_dims      = std::stoi(command_args[9]);  // Matrix dimensions
                    
                    // Parse shard_index_override from the command string
                    int shard_index_override = -1;
                    if (command_args.size() > 10) {
                        shard_index_override = std::stoi(command_args[10]);
                    }

                    // ============================================================
                    // PARSE SEND_BACK STRING FOR HIERARCHICAL SPLIT INFORMATION
                    // ============================================================
                    int send_back_number = 0;
                    std::vector<int> hierarchical_split_order;
                    
                    // Check if format contains '/' (new hierarchical format: "4/011" or "-4/011")
                    size_t slash_pos = send_back_str.find('/');
                    if (slash_pos != std::string::npos) {
                        // New format: "4/011" or "-4/011"
                        std::string total_shards_str = send_back_str.substr(0, slash_pos);
                        std::string split_info_str = send_back_str.substr(slash_pos + 1);
                        
                        // Parse total number of shards
                        send_back_number = std::stoi(total_shards_str);
                        
                        // Parse hierarchical split order from split_info_str
                        if (!split_info_str.empty()) {
                            // String format: first char is initial split dim, rest are hierarchical order
                            // Example: "011" means initial dim=0, then splits: [1, 1]
                            for (size_t i = 0; i < split_info_str.size(); i++) {
                                if (split_info_str[i] == '0' || split_info_str[i] == '1') {
                                    hierarchical_split_order.push_back(split_info_str[i] - '0');
                                }
                            }
                            
                            std::cout << "DEBUG: Parsed hierarchical info - total_shards=" << send_back_number 
                                    << ", split_order=[";
                            for (size_t i = 0; i < hierarchical_split_order.size(); i++) {
                                if (i > 0) std::cout << ",";
                                std::cout << hierarchical_split_order[i];
                            }
                            std::cout << "]" << std::endl;
                        }
                    } else {
                        // Old format: just number (no hierarchical splits)
                        send_back_number = std::stoi(send_back_str);
                    }
                    
                    // Store for later use in result distribution
                    send_back_number_of_shards = send_back_number;
                    
                    bool operation_success = false;
                    std::string backend_name;

                    // Dispatch to unified matrix operation function
                    if (command_type == "llama")
                    {
                        operation_success = matrix_operation(
                            command_type,
                            command_args[3].c_str(),   // Matrix B path
                            transposeB,
                            command_args[1].c_str(),   // Matrix A path
                            transposeA,
                            use_gpu,
                            gpu_id,
                            send_back_str,  // Pass the full string (e.g., "4/011")
                            operation_type,
                            n_dims,
                            shard_index_override,
                            hierarchical_split_order  // Pass parsed hierarchical split order
                        );
                    }
                    else
                    {
                        operation_success = matrix_operation(
                            command_type,
                            command_args[1].c_str(),   // Matrix A path
                            transposeA,
                            command_args[3].c_str(),   // Matrix B path
                            transposeB,
                            use_gpu,
                            gpu_id,
                            send_back_str,  // Pass the full string (e.g., "4/011")
                            operation_type,
                            n_dims,
                            shard_index_override,
                            hierarchical_split_order  // Pass parsed hierarchical split order
                        );
                    }
                    
                    if (command_type == "llama")
                        backend_name = "LLaMA/Vulkan";
                    else if (command_type == "torch")
                        backend_name = "PyTorch";
                    else if (command_type == "opencl")
                        backend_name = "OpenCL";

                    // Report operation outcome
                    if (operation_success) {
                        std::cout << "✅ " << backend_name << " operation completed successfully" << std::endl;
                        std::cout << "   • Operation: " << operation_type << std::endl;
                        std::cout << "   • GPU: " << (use_gpu ? "Yes (ID: " + std::to_string(gpu_id) + ")" : "No") << std::endl;
                        std::cout << "   • Result shards: " << send_back_number << std::endl;
                        if (!hierarchical_split_order.empty()) {
                            std::cout << "   • Hierarchical splits: " << hierarchical_split_order.size() << " levels" << std::endl;
                        }
                        return 0;
                    } else {
                        std::cerr << "❌ " << backend_name << " operation failed: " << operation_type << std::endl;
                        return -7;
                    }

                } 
                else 
                {
                    std::cerr << "❌ Unsupported server command type: '" << command_type << "'" << std::endl;
                    std::cerr << "   Supported commands: llama, opencl, torch" << std::endl;
                    return -6;
                }

            } 
            catch (const std::exception& e) 
            {
                std::cerr << "❌ Error executing server command: " << e.what() << std::endl;
                std::cerr << "   Command: " << command << std::endl;
                return -1;
            }
        }

        std::pair<std::string, int> get_matrix_name_and_shard_number(const std::string& shard_path) 
        {
            // Extract just the filename from the full path (remove directory portion)
            // Example: "/path/to/matrixA_shard_42.bin" -> "matrixA_shard_42.bin"
            std::string filename = shard_path.substr(shard_path.find_last_of("/") + 1);
            
            // Look for the shard pattern in the filename
            // Shard files follow the naming convention: <matrix_name>_shard_<number>.<extension>
            auto find_shard_pos = [](const std::string& name) -> std::pair<size_t, size_t> {
                size_t pos = name.find("_shard_");
                if (pos != std::string::npos) {
                    return {pos, 7};  // length of "_shard_"
                }
                // Accept legacy/pluralized variant to stay compatible with older runs
                pos = name.find("_shards_");
                if (pos != std::string::npos) {
                    return {pos, 8};  // length of "_shards_"
                }
                return {std::string::npos, 0};
            };

            auto [shard_pos, shard_token_len] = find_shard_pos(filename);
            
            if (shard_pos != std::string::npos) {
                // Extract the base matrix name (everything before the shard token)
                std::string matrix_name = filename.substr(0, shard_pos);
                
                // Extract the shard number portion (everything after the shard token)
                std::string shard_part = filename.substr(shard_pos + shard_token_len);
                
                // Remove file extension if present to isolate just the number
                // Example: "42.bin" -> "42"
                size_t dot_pos = shard_part.find_last_of(".");
                if (dot_pos != std::string::npos) {
                    shard_part = shard_part.substr(0, dot_pos);
                }
                
                try {
                    // Convert the shard number string to integer
                    int shard_number = std::stoi(shard_part);
                    
                    // Return both the matrix name and shard number
                    return {matrix_name, shard_number};
                } catch (const std::exception& e) {
                    // Handle parsing errors (e.g., if shard_part contains non-numeric characters)
                    // Return -1 as invalid shard number to indicate parsing failure
                    std::cerr << "⚠️ Failed to parse shard number from: '" << shard_part 
                            << "' in file: " << filename << std::endl;
                    return {matrix_name, -1};
                }
            }
            
            // If the filename doesn't match the shard pattern, return the entire filename
            // as the matrix name and -1 to indicate this is not a shard file
            // This handles cases like regular matrix files or incorrectly named files
            return {filename, -1};
        }

        void save_file_handler()
        {
            // Move reserved files to local copy under lock for processing
            std::vector<ReservedFiles> local_reserved_files;

            {
                std::lock_guard<std::mutex> lock(file_data_mutex);

                if (reserved_files_list.empty())
                {
                    std::cout << "No files to save" << std::endl;
                    return;
                }

                local_reserved_files = std::move(reserved_files_list);
                reserved_files_list.clear();

                std::cout << "Processing: " << local_reserved_files.size() << " reserved file(s)" << std::endl;
            }

            // Iterate through each reserved file entry and handle cases:
            // - parallel (ETH + WiFi halves)
            // - single-interface ETH
            // - single-interface WiFi
            std::string tmp_file_name = "";
            for (auto &rf : local_reserved_files)
            {
                std::string filename = rf.save_parallel_file_name.empty() ? std::string("unknown") : rf.save_parallel_file_name[0];
                // Helper lambda to write raw bytes to path
                tmp_file_name = filename;
                auto write_raw = [&](const std::filesystem::path &path, const std::vector<uint8_t> &bytes) -> bool {
                    std::filesystem::create_directories(path.parent_path());
                    std::ofstream file(path, std::ios::binary);
                    if (!file.is_open()) return false;
                    file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
                    file.close();
                    return true;
                };

                // If we have both halves (parallel) -> combine
                if ((rf.is_parallel) || (!rf.received_data_eth_file.empty() && !rf.received_data_wifi_file.empty()))
                {
                    std::vector<uint8_t> combined;
                    combined.reserve(rf.received_data_eth_file.size() + rf.received_data_wifi_file.size());
                    combined.insert(combined.end(), rf.received_data_eth_file.begin(), rf.received_data_eth_file.end());
                    combined.insert(combined.end(), rf.received_data_wifi_file.begin(), rf.received_data_wifi_file.end());

                    size_t sent_back_pos = filename.find("sent_back=");
                    if (sent_back_pos != std::string::npos)
                    {
                        std::string actual_filename = filename.substr(sent_back_pos + 10);
                        std::filesystem::path save_path = std::filesystem::path(matrix_results_folder) / actual_filename;
                        if (write_raw(save_path, combined))
                            std::cout << "PARALLEL saved to RESULTS: " << save_path << " (" << combined.size() << " bytes)" << std::endl;
                        else
                            std::cerr << "Failed to save PARALLEL sent_back: " << save_path << std::endl;

                        // Head node-specific processing for combined sent_back (attempt to parse 4D tensor)
                        if (local_IP_eth == head_node_ip_eth)
                        {
                            const uint8_t* p = combined.data();
                            int ndim = *reinterpret_cast<const int*>(p);
                            p += sizeof(int);
                            if (ndim == 4)
                            {
                                int dims[4];
                                for (int i = 0; i < 4; ++i) { dims[i] = *reinterpret_cast<const int*>(p); p += sizeof(int); }
                                int batch = dims[0];
                                int depth = dims[1];
                                int rows = dims[2];
                                int cols = dims[3];
                                size_t total_elements = static_cast<size_t>(batch) * depth * rows * cols;
                                auto shard_data = std::make_unique<float[]>(total_elements);
                                std::memcpy(shard_data.get(), p, total_elements * sizeof(float));

                                handle_combine_matrix_shard_list(actual_filename, std::move(shard_data), rows, cols, 0, {}); 
                                // move this call to 'run_server_command' function 
                            }
                        }
                    }
                    else
                    {
                        std::filesystem::path save_path = std::filesystem::path(matrix_shard_folder) / filename;
                        // Try to validate as 4D binary and save via save_matrix_bin; otherwise write raw
                        bool saved = false;
                        if (combined.size() >= static_cast<size_t>(5 * sizeof(int)))
                        {
                            const uint8_t* p = combined.data();
                            int ndim = *reinterpret_cast<const int*>(p);
                            if (ndim == 4)
                            {
                                MatrixResult result;
                                result.dims[0] = *reinterpret_cast<const int*>(p + sizeof(int));
                                result.dims[1] = *reinterpret_cast<const int*>(p + 2 * sizeof(int));
                                result.dims[2] = *reinterpret_cast<const int*>(p + 3 * sizeof(int));
                                result.dims[3] = *reinterpret_cast<const int*>(p + 4 * sizeof(int));
                                size_t total_elements = static_cast<size_t>(result.dims[0]) * result.dims[1] * result.dims[2] * result.dims[3];
                                result.data = std::make_unique<float[]>(total_elements);
                                std::memcpy(result.data.get(), p + 5 * sizeof(int), total_elements * sizeof(float));
                                if (save_matrix_bin(save_path.c_str(), result))
                                {
                                    saved = true;
                                    std::cout << "PARALLEL saved to SHARDS: " << save_path << " (" << combined.size() << " bytes)" << std::endl;
                                }
                            }
                        }

                        if (!saved)
                        {
                            if (write_raw(save_path, combined))
                                std::cout << "PARALLEL saved (raw): " << save_path << " (" << combined.size() << " bytes)" << std::endl;
                            else
                                std::cerr << "Failed to save PARALLEL file: " << save_path << std::endl;
                        }
                    }
                }
                // ETH single-interface file
                else if (!rf.received_data_eth_file.empty())
                {
                    const auto &data = rf.received_data_eth_file;
                    size_t sent_back_pos = filename.find("sent_back=");
                    if (sent_back_pos != std::string::npos)
                    {
                        std::string actual_filename = filename.substr(sent_back_pos + 10);
                        std::filesystem::path save_path = std::filesystem::path(matrix_results_folder) / actual_filename;
                        if (write_raw(save_path, data))
                        {
                            std::cout << "ETH sent_back saved to RESULTS: " << save_path << " (" << data.size() << " bytes)" << std::endl;
                        }
                        else
                        {
                            std::cerr << "Failed to save ETH sent_back: " << save_path << std::endl;
                        }

                        // Head node-specific processing for shard combination
                        if (local_IP_eth == head_node_ip_eth)
                        {
                            const uint8_t* p = data.data();
                            int ndim = *reinterpret_cast<const int*>(p);
                            p += sizeof(int);
                            if (ndim == 4)
                            {
                                int dims[4];
                                for (int i = 0; i < 4; ++i) { dims[i] = *reinterpret_cast<const int*>(p); p += sizeof(int); }
                                int batch = dims[0];
                                int depth = dims[1];
                                int rows = dims[2];
                                int cols = dims[3];
                                size_t total_elements = static_cast<size_t>(batch) * depth * rows * cols;
                                auto shard_data = std::make_unique<float[]>(total_elements);
                                std::memcpy(shard_data.get(), p, total_elements * sizeof(float));

                                handle_combine_matrix_shard_list(actual_filename, std::move(shard_data), rows, cols, 0, {});
                            }
                        }
                    }
                    else
                    {
                        // Regular ETH file: validate 4D and save via save_matrix_bin
                        std::filesystem::path save_path = std::filesystem::path(matrix_shard_folder) / filename;
                        if (data.size() >= static_cast<size_t>(5 * sizeof(int)))
                        {
                            const uint8_t* p = data.data();
                            int ndim = *reinterpret_cast<const int*>(p);
                            if (ndim != 4)
                            {
                                std::cerr << "ERROR: Worker sent non-4D tensor: " << filename << " (ndim=" << ndim << ")" << std::endl;
                            }
                            else
                            {
                                MatrixResult result;
                                result.dims[0] = *reinterpret_cast<const int*>(p + sizeof(int));
                                result.dims[1] = *reinterpret_cast<const int*>(p + 2 * sizeof(int));
                                result.dims[2] = *reinterpret_cast<const int*>(p + 3 * sizeof(int));
                                result.dims[3] = *reinterpret_cast<const int*>(p + 4 * sizeof(int));
                                size_t total_elements = static_cast<size_t>(result.dims[0]) * result.dims[1] * result.dims[2] * result.dims[3];
                                result.data = std::make_unique<float[]>(total_elements);
                                std::memcpy(result.data.get(), p + 5 * sizeof(int), total_elements * sizeof(float));

                                if (save_matrix_bin(save_path.c_str(), result))
                                    std::cout << "ETH saved to SHARDS: " << save_path << " (" << data.size() << " bytes)" << std::endl;
                                else
                                    std::cerr << "Failed to save ETH file: " << save_path << std::endl;
                            }
                        }
                        else
                        {
                            if (write_raw(save_path, data))
                                std::cout << "ETH saved (raw) to SHARDS: " << save_path << " (" << data.size() << " bytes)" << std::endl;
                            else
                                std::cerr << "Failed to save ETH file: " << save_path << std::endl;
                        }
                    }
                }
                // WiFi single-interface file
                else if (!rf.received_data_wifi_file.empty())
                {
                    const auto &data = rf.received_data_wifi_file;
                    size_t sent_back_pos = filename.find("sent_back=");
                    if (sent_back_pos != std::string::npos)
                    {
                        std::string actual_filename = filename.substr(sent_back_pos + 10);
                        std::filesystem::path save_path = std::filesystem::path(matrix_results_folder) / actual_filename;
                        if (write_raw(save_path, data))
                            std::cout << "WiFi sent_back saved to RESULTS: " << save_path << " (" << data.size() << " bytes)" << std::endl;
                        else
                            std::cerr << "Failed to save WiFi sent_back: " << save_path << std::endl;
                    }
                    else
                    {
                        std::filesystem::path save_path = std::filesystem::path(matrix_shard_folder) / filename;
                        if (write_raw(save_path, data))
                            std::cout << "WiFi saved to SHARDS: " << save_path << " (" << data.size() << " bytes)" << std::endl;
                        else
                            std::cerr << "Failed to save WiFi file: " << save_path << std::endl;
                    }
                }
                else
                {
                    std::cout << "Skipping empty ReservedFiles entry for: " << filename << std::endl;
                }
            }

            // Send acknowledgment if this is not the head node
            if (local_IP_eth != head_node_ip_eth)
            {
                //std::cout << tmp_file_name;
                send_ack(tmp_file_name);
            }

            std::cout << "Save file handler completed" << std::endl;
        }

        bool send_back_file(const std::string& local_file_path,
                            const std::string& filename,
                            MatrixResult& save_result,
                            int total_shards,
                            const std::vector<int>& hierarchical_split_order,  // NEW parameter
                            const std::string& selected_backend)
        {
            std::cout << "SENDING BACK FILE" << std::endl;
            bool is_head_node = (local_IP_eth == head_node_ip_eth || local_IP_wifi == head_node_ip_wifi);

            // ============================================================
            // WORKER NODE → SEND RESULT BACK TO HEAD (HEAD-SEMANTIC FORMAT)
            // ============================================================
            if (!is_head_node)
            {
                std::string send_back_filename = "sent_back=" + filename;
                std::cout << "Worker sending result back to head: " << send_back_filename << std::endl;

                std::vector<uint8_t> buffer;

                // Network contract: Always send logical 4D tensor
                int ndim = 4;
                buffer.insert(buffer.end(),
                    reinterpret_cast<uint8_t*>(&ndim),
                    reinterpret_cast<uint8_t*>(&ndim) + sizeof(int));

                // Normalize dimensions according to backend format
                int batch, depth, shard_rows, shard_cols;
                if (selected_backend == "llama")  // for incase things go fucked does nothing
                {  
                    // GGML format: {cols, rows, depth, batch}
                    batch      = save_result.dims[0];  // batch is index 3
                    depth      = save_result.dims[1];  // depth is index 2
                    shard_rows = save_result.dims[2];  // rows is index 1
                    shard_cols = save_result.dims[3];  // cols is index 0
                }  
                else if (selected_backend == "torch")  // for incase things go fucked does nothing 
                {  
                    // Torch format: {batch, depth, rows, cols}
                    batch      = save_result.dims[0];
                    depth      = save_result.dims[1];
                    shard_rows = save_result.dims[2];
                    shard_cols = save_result.dims[3];
                }

                // IMPORTANT: save_result.dims[] must already reflect the logical shape
                // If not, this indicates a backend issue, not a network issue
                int dims[4] = { batch, depth, shard_rows, shard_cols };

                // Insert all 4 dimensions into buffer
                for (int i = 0; i < 4; i++)
                {
                    buffer.insert(buffer.end(),
                        reinterpret_cast<uint8_t*>(&dims[i]),
                        reinterpret_cast<uint8_t*>(&dims[i]) + sizeof(int));
                }

                // Calculate and insert data payload
                size_t total_elements =
                    static_cast<size_t>(batch) *
                    static_cast<size_t>(depth) *
                    static_cast<size_t>(shard_rows) *
                    static_cast<size_t>(shard_cols);

                buffer.insert(buffer.end(),
                    reinterpret_cast<uint8_t*>(save_result.data.get()),
                    reinterpret_cast<uint8_t*>(save_result.data.get()) +
                    total_elements * sizeof(float));

                // Send data to head node via ZeroMQ
                zmq::message_t filename_msg(send_back_filename.data(), send_back_filename.size());
                zmq::message_t data_msg(buffer.data(), buffer.size());

                head_node_sender_eth.send(filename_msg, zmq::send_flags::sndmore);
                head_node_sender_eth.send(data_msg, zmq::send_flags::none);

                std::cout << "Result sent to head node: "
                        << send_back_filename << " (" << buffer.size() << " bytes)" << std::endl;

                return true;
            }

            // ============================================================
            // HEAD NODE → SAVE FILE + TRACK SHARDS
            // ============================================================
            if (is_head_node)
            {
                // Extract shard dimensions
                int shard_rows = save_result.dims[2];
                int shard_cols = save_result.dims[3];
                size_t data_size = shard_rows * shard_cols * sizeof(float);

                // Copy data into unique_ptr for shard processing
                auto shard_data = std::make_unique<float[]>(shard_rows * shard_cols);
                std::memcpy(shard_data.get(),
                            save_result.data.get(),
                            data_size);

                // Process shard through combination handler WITH hierarchical split order
                bool result = handle_combine_matrix_shard_list(
                    filename,
                    std::move(shard_data),
                    shard_rows,
                    shard_cols,
                    total_shards,
                    hierarchical_split_order  // PASS hierarchical split order
                );

                std::cout << "Head node processed shard: " << filename 
                        << " (" << data_size << " bytes)";
                
                if (!hierarchical_split_order.empty()) {
                    std::cout << " with hierarchical splits [";
                    for (size_t i = 0; i < hierarchical_split_order.size(); i++) {
                        if (i > 0) std::cout << ",";
                        std::cout << hierarchical_split_order[i];
                    }
                    std::cout << "]";
                }
                std::cout << std::endl;

                return result;
            }

            return false;
        }

        bool handle_combine_matrix_shard_list(  
            const std::string& filename,  
            std::unique_ptr<float[]> data,  
            int shard_rows,  
            int shard_cols,  
            int total_shards,
            const std::vector<int>& hierarchical_split_order = {}  // NEW parameter
        )  
        {  
            // ============================================================
            // DEBUG: Print incoming parameters
            // ============================================================
            std::cout << "DEBUG: handle_combine_matrix_shard_list called" << std::endl;
            std::cout << "DEBUG: filename='" << filename << "'" << std::endl;
            std::cout << "DEBUG: shard_rows=" << shard_rows << ", shard_cols=" << shard_cols << std::endl;
            std::cout << "DEBUG: total_shards=" << total_shards << std::endl;
            std::cout << "DEBUG: hierarchical_split_order size=" << hierarchical_split_order.size() << std::endl;
            
            // Print the actual values of hierarchical_split_order
            if (!hierarchical_split_order.empty()) {
                std::cout << "DEBUG: hierarchical_split_order values: [";
                for (size_t i = 0; i < hierarchical_split_order.size(); i++) {
                    if (i > 0) std::cout << ", ";
                    std::cout << hierarchical_split_order[i];
                }
                std::cout << "]" << std::endl;
            }
            
            // ============================================================
            // EXTRACT MATRIX NAME AND SHARD NUMBER
            // ============================================================
            auto [matrix_name, shard_num] = get_matrix_name_and_shard_number(filename);  
            std::cout << "DEBUG: Extracted matrix_name='" << matrix_name 
                    << "', shard_num=" << shard_num << std::endl;

            // ============================================================
            // BUILD SHARD BYTES WITH METADATA
            // ============================================================
            std::vector<uint8_t> shard_bytes;  

            // Always use 4D format: batch, depth, rows, cols
            int ndim = 4; 
            shard_bytes.insert(  
                shard_bytes.end(),  
                reinterpret_cast<uint8_t*>(&ndim),  
                reinterpret_cast<uint8_t*>(&ndim) + sizeof(int)  
            );  

            // Dimensions for a single shard: batch=1, depth=1, rows, cols
            int dims[4] = {1, 1, shard_rows, shard_cols};  
            for (int i = 0; i < 4; ++i)  
            {  
                shard_bytes.insert(  
                    shard_bytes.end(),  
                    reinterpret_cast<uint8_t*>(&dims[i]),  
                    reinterpret_cast<uint8_t*>(&dims[i]) + sizeof(int)  
                );  
            }  

            // Append actual matrix data
            size_t data_size = static_cast<size_t>(shard_rows) * shard_cols * sizeof(float);  
            shard_bytes.insert(  
                shard_bytes.end(),  
                reinterpret_cast<uint8_t*>(data.get()),  
                reinterpret_cast<uint8_t*>(data.get()) + data_size  
            );  

            std::cout << "DEBUG: Created shard_bytes of size " << shard_bytes.size() << std::endl;

            // ============================================================
            // TRACK SHARD (check if we're already collecting for this matrix)
            // ============================================================
            std::cout << "DEBUG: Checking " << combined_matrix_shards_list.size() 
                    << " existing tracking entries" << std::endl;
            
            for (auto& combined : combined_matrix_shards_list)  
            {  
                // Extract base name from the tracked entry
                auto [combined_name, _] = get_matrix_name_and_shard_number(combined.file_name);  
                std::cout << "DEBUG: Checking against entry with file_name='" << combined.file_name 
                        << "', extracted name='" << combined_name << "'" << std::endl;
                
                // Found existing entry for this matrix
                if (combined_name == matrix_name)  
                {  
                    std::cout << "DEBUG: FOUND MATCH for matrix '" << matrix_name << "'" << std::endl;
                    std::cout << "DEBUG: Current shard count: " << combined.total_shards_reserved 
                            << " of " << combined.number_of_shards_needed << std::endl;
                    
                    // Update tracking information
                    combined.total_shards_reserved++;  
                    combined.shard_numbers.push_back(shard_num);  
                    combined.received_matrix_data.push_back(std::move(shard_bytes));  
                    
                    // Store dimensions for this shard
                    std::vector<int> shard_dims = {1, 1, shard_rows, shard_cols};  
                    combined.dims_list.push_back(shard_dims);  

                    std::cout << "DEBUG: Updated shard count to: " << combined.total_shards_reserved 
                            << " of " << combined.number_of_shards_needed << std::endl;

                    // ============================================================
                    // CHECK IF ALL SHARDS HAVE ARRIVED
                    // ============================================================
                    // Use ABSOLUTE value for comparison since total_shards_reserved is always positive
                    int needed_shards_absolute = std::abs(combined.number_of_shards_needed);
                    
                    if (combined.total_shards_reserved == needed_shards_absolute)  
                    {  
                        // Determine system from the SIGN of number_of_shards_needed
                        bool is_system2 = (combined.number_of_shards_needed < 0);
                        bool is_system3 = (!combined.hierarchical_split_order.empty());  // NEW: Check for hierarchical splits
                        
                        std::cout << "DEBUG: ALL SHARDS RECEIVED! Triggering combine..." << std::endl;
                        std::cout << "All " << needed_shards_absolute 
                                << " shards received. Combining matrix: " << matrix_name 
                                << " (System " << (is_system2 ? "2" : "1") << ")" 
                                << (is_system3 ? " with hierarchical splits" : "") << std::endl;  

                        MatrixResult full;  
                        
                        // ============================================================
                        // DETERMINE WHICH COMBINATION METHOD TO USE
                        // ============================================================
                        // System 3 (hierarchical) takes priority if present
                        // Then System 2 (grid assembly for A @ B.T)
                        // Finally System 1 (standard concatenation)
                        // ============================================================
                        
                        /*
                        if (is_system3)
                        {
                            // System 3: Hierarchical combination
                            std::cout << "DEBUG: Calling System 3 hierarchical combination" << std::endl;
                            std::cout << "DEBUG: Split order: [";
                            for (size_t i = 0; i < combined.hierarchical_split_order.size(); i++) {
                                if (i > 0) std::cout << ", ";
                                std::cout << combined.hierarchical_split_order[i];
                            }
                            std::cout << "]" << std::endl;
                            
                            // FIXED: Pass combined.hierarchical_split_order, not the parameter
                            full = combine_hierarchical_results(combined, combined.hierarchical_split_order);
                            
                            // If hierarchical combination fails, fall back to appropriate system
                            if (!full.data) {
                                std::cout << "WARNING: System 3 hierarchical combination failed, falling back..." << std::endl;
                                if (is_system2) {
                                    std::cout << "DEBUG: Falling back to System 2 grid assembly" << std::endl;
                                    full = combine_matrix_shards_grid_2d(combined);  
                                } else {
                                    std::cout << "DEBUG: Falling back to System 1 concatenation" << std::endl;
                                    full = combine_matrix_shards_2d(combined);  
                                }
                            }
                        }
                        */
                        if (is_system2) {
                            // System 2: Grid assembly (GEMM results)
                            std::cout << "DEBUG: Calling System 2 grid assembly" << std::endl;
                            full = combine_matrix_shards_grid_2d(combined);  
                        } 
                        else {
                            // System 1: Standard concatenation  
                            std::cout << "DEBUG: Calling System 1 concatenation" << std::endl;
                            full = combine_matrix_shards_2d(combined);  
                        }

                        if (!full.data)  
                        {  
                            std::cerr << "ERROR: Failed to combine matrix shards for: " << matrix_name << std::endl;  
                        }  
                        else  
                        {  
                            // Save combined matrix to file
                            std::string final_path =  
                                std::filesystem::path(matrix_shard_folder) /  
                                (combined_name + "_combined.bin");  

                            std::cout << "DEBUG: Saving combined matrix to: " << final_path << std::endl;
                            save_matrix_bin(final_path.c_str(), full);

                            //wait_for_acks(1, "ACK_matrixOp_complete_CONFIRM");
                            send_ack("ACK_combined_matrix_saved");
                              

                            std::cout << "Combined matrix saved: " << final_path << std::endl;  
                        }  

                        // Remove completed entry from tracking list
                        std::cout << "DEBUG: Removing completed entry from tracking list" << std::endl;
                        combined_matrix_shards_list.erase(  
                            std::remove_if(  
                                combined_matrix_shards_list.begin(),  
                                combined_matrix_shards_list.end(),  
                                [&](const combined_matrix_shards& c)  
                                {  
                                    auto [n, __] = get_matrix_name_and_shard_number(c.file_name);  
                                    return n == combined_name;  
                                }),  
                            combined_matrix_shards_list.end()  
                        );  
                    }  
                    send_ack("ACK_combined_matrix_saved");
                    return true;  
                }  
            }  

            // ============================================================
            // FIRST SHARD FOR THIS MATRIX (create new tracking entry)
            // ============================================================
            std::cout << "DEBUG: No existing entry found. Creating new tracking entry." << std::endl;
            
            combined_matrix_shards combined;  
            combined.file_name = matrix_name;  
            combined.number_of_shards_needed = total_shards;  // Store signed value (-6 for System 2)
            combined.total_shards_reserved = 1;  
            combined.shard_numbers.push_back(shard_num);  
            combined.received_matrix_data.push_back(std::move(shard_bytes));  
            
            // Store dimensions for the first shard  
            std::vector<int> shard_dims = {1, 1, shard_rows, shard_cols};  
            combined.dims_list.push_back(shard_dims);  
            
            // Store hierarchical split order if provided
            combined.hierarchical_split_order = hierarchical_split_order;

            // Add new entry to tracking list
            combined_matrix_shards_list.push_back(std::move(combined));  

            // Determine system type for logging
            bool is_system2 = (total_shards < 0);
            int absolute_shard_count = std::abs(total_shards);
            
            std::cout << "Started tracking new matrix: " << matrix_name 
                    << " (shard " << shard_num << " of " << absolute_shard_count 
                    << ", System " << (is_system2 ? "2" : "1") << ")";
            
            if (!hierarchical_split_order.empty()) {
                std::cout << " with hierarchical splits: [";
                for (size_t i = 0; i < hierarchical_split_order.size(); i++) {
                    if (i > 0) std::cout << ", ";
                    std::cout << hierarchical_split_order[i];
                }
                std::cout << "]";
            }
            std::cout << std::endl;
            
            return true;  
        }

        MatrixResult combine_matrix_shards_2d(const combined_matrix_shards& combined)
        {
            MatrixResult result;

            // Early return if no data received
            if (combined.received_matrix_data.empty()) {
                return result;
            }

            // ============================================================
            // STEP 1: SORT SHARDS BY SHARD NUMBER
            // ============================================================
            // Create vector of (shard_number, shard_data_pointer, dims) pairs
            struct ShardInfo {
                int number;
                const std::vector<uint8_t>* data;
                std::vector<int> dims;
            };
            
            std::vector<ShardInfo> sorted_shards;
            
            auto shard_num_it = combined.shard_numbers.begin();
            auto data_it      = combined.received_matrix_data.begin();
            auto dims_it      = combined.dims_list.begin();

            // Pair each shard number with its corresponding data and dimensions
            for (; shard_num_it != combined.shard_numbers.end() &&
                data_it      != combined.received_matrix_data.end() &&
                dims_it      != combined.dims_list.end();
                ++shard_num_it, ++data_it, ++dims_it)
            {
                sorted_shards.push_back({*shard_num_it, &(*data_it), *dims_it});
            }

            // Sort by shard number to ensure correct concatenation order
            std::sort(sorted_shards.begin(), sorted_shards.end(),
                    [](auto& a, auto& b){ return a.number < b.number; });

            // ============================================================
            // STEP 2: COMPUTE TOTAL DIMENSIONS BASED ON JOIN DIMENSION
            // ============================================================
            int total_rows = 0;
            int total_cols = 0;
            int join_dim = combined.join_dim; // 0 for rows, 1 for columns

            if (join_dim == 0) {
                // Concatenate along rows (vertical stacking)
                for (const auto& shard : sorted_shards) {
                    int rows = shard.dims[2];  // rows dimension
                    int cols = shard.dims[3];  // cols dimension
                    total_rows += rows;
                    total_cols = std::max(total_cols, cols);
                }
                std::cout << "Combining " << sorted_shards.size() << " shards along rows into "
                        << total_rows << "x" << total_cols << " matrix" << std::endl;
            } else if (join_dim == 1) {
                // Concatenate along columns (horizontal stacking)
                for (const auto& shard : sorted_shards) {
                    int rows = shard.dims[2];  // rows dimension
                    int cols = shard.dims[3];  // cols dimension
                    total_rows = std::max(total_rows, rows);
                    total_cols += cols;
                }
                std::cout << "Combining " << sorted_shards.size() << " shards along columns into "
                        << total_rows << "x" << total_cols << " matrix" << std::endl;
            } else {
                std::cerr << "Error: Invalid join_dim " << join_dim 
                        << ". Must be 0 (rows) or 1 (columns)" << std::endl;
                return result;
            }

            // ============================================================
            // STEP 3: ALLOCATE OUTPUT MATRIX
            // ============================================================
            result.dims[0] = 1;  // batch
            result.dims[1] = 1;  // depth
            result.dims[2] = total_rows;
            result.dims[3] = total_cols;

            // Allocate memory for combined matrix
            result.data = std::make_unique<float[]>(
                static_cast<size_t>(total_rows) * total_cols
            );

            // Initialize with zeros
            std::fill(result.data.get(), 
                    result.data.get() + total_rows * total_cols, 
                    0.0f);

            // ============================================================
            // STEP 4: COPY SHARDS BASED ON JOIN DIMENSION
            // ============================================================
            if (join_dim == 0) {
                // Concatenate along rows (vertical stacking)
                int row_offset = 0;  // Track where to place next shard vertically
                
                for (const auto& shard : sorted_shards) {
                    // Get shard data pointer (skip metadata)
                    const uint8_t* p = shard.data->data();
                    p += sizeof(int);  // skip ndim
                    p += sizeof(int) * shard.dims.size();  // skip dims
                    
                    int rows = shard.dims[2];
                    int cols = shard.dims[3];
                    const float* shard_data = reinterpret_cast<const float*>(p);
                    
                    std::cout << "  Copying shard " << shard.number << ": " 
                            << rows << "x" << cols << " at row offset " << row_offset << std::endl;
                    
                    // Row-major copy: concatenate vertically
                    for (int r = 0; r < rows; ++r) {
                        std::memcpy(
                            result.data.get() + (row_offset + r) * total_cols,
                            shard_data + r * cols,
                            sizeof(float) * cols
                        );
                    }
                    
                    row_offset += rows;
                }
            } else {
                // Concatenate along columns (horizontal stacking)
                int col_offset = 0;  // Track where to place next shard horizontally
                
                for (const auto& shard : sorted_shards) {
                    // Get shard data pointer (skip metadata)
                    const uint8_t* p = shard.data->data();
                    p += sizeof(int);  // skip ndim
                    p += sizeof(int) * shard.dims.size();  // skip dims
                    
                    int rows = shard.dims[2];
                    int cols = shard.dims[3];
                    const float* shard_data = reinterpret_cast<const float*>(p);
                    
                    std::cout << "  Copying shard " << shard.number << ": " 
                            << rows << "x" << cols << " at column offset " << col_offset << std::endl;
                    
                    // Row-major copy: concatenate horizontally
                    for (int r = 0; r < rows; ++r) {
                        std::memcpy(
                            result.data.get() + r * total_cols + col_offset,
                            shard_data + r * cols,
                            sizeof(float) * cols
                        );
                    }
                    
                    col_offset += cols;
                }
            }

            std::cout << "Matrix combination complete (joined along dim=" << join_dim << ")" << std::endl;
            
            return result;
        }

        MatrixResult combine_matrix_shards_grid_2d(const combined_matrix_shards& combined)
        {
            MatrixResult result;

            if (combined.received_matrix_data.empty()) {
                return result;
            }

            std::cout << "DEBUG: Starting System 2 grid combine" << std::endl;
            std::cout << "DEBUG: Total shards: " << combined.received_matrix_data.size() << std::endl;

            // ============================================================
            // STEP 1: PAIR SHARD NUMBERS WITH DATA AND SORT
            // ============================================================
            std::vector<std::pair<int, const std::vector<uint8_t>*>> sorted_shards;

            auto shard_num_it = combined.shard_numbers.begin();
            auto data_it      = combined.received_matrix_data.begin();

            for (; shard_num_it != combined.shard_numbers.end() &&
                data_it      != combined.received_matrix_data.end();
                ++shard_num_it, ++data_it)
            {
                sorted_shards.emplace_back(*shard_num_it, &(*data_it));
            }

            std::sort(sorted_shards.begin(), sorted_shards.end(),
                    [](auto& a, auto& b){ return a.first < b.first; });

            int total_shards = sorted_shards.size();

            // ============================================================
            // STEP 2: SYSTEM-2 GRID GEOMETRY
            // ============================================================
            if (total_shards % 2 != 0) {
                std::cout << "ERROR: System 2 requires even shard count\n";
                return combine_matrix_shards_2d(combined);
            }

            int grid_rows = 2;                 // A is always split into 2
            int grid_cols = total_shards / 2;  // B shards

            std::cout << "DEBUG: Grid " << grid_rows << " × " << grid_cols << std::endl;

            // ============================================================
            // STEP 3: READ FIRST SHARD DIMENSIONS
            // ============================================================
            const uint8_t* p = sorted_shards[0].second->data();
            int ndim = *reinterpret_cast<const int*>(p);
            p += sizeof(int);

            std::vector<int> first_dims(ndim);
            for (int i = 0; i < ndim; ++i) {
                first_dims[i] = *reinterpret_cast<const int*>(p);
                p += sizeof(int);
            }

            int shard_rows = first_dims[2];

            // ============================================================
            // STEP 4: DETERMINE COLUMN WIDTHS (SUPPORT UNEVEN B SPLITS)
            // ============================================================
            std::vector<int> col_dims(grid_cols, 0);

            for (const auto& [shard_num, shard_bytes] : sorted_shards) {

                int col_idx = shard_num % grid_cols;

                const uint8_t* p2 = shard_bytes->data();
                p2 += sizeof(int); // ndim

                std::vector<int> dims(ndim);
                for (int i = 0; i < ndim; ++i) {
                    dims[i] = *reinterpret_cast<const int*>(p2);
                    p2 += sizeof(int);
                }

                if (col_dims[col_idx] == 0) {
                    col_dims[col_idx] = dims[3];
                }
            }

            int total_rows = grid_rows * shard_rows;
            int total_cols = 0;
            for (int w : col_dims) total_cols += w;

            std::cout << "DEBUG: Final matrix " << total_rows << " × " << total_cols << std::endl;

            // ============================================================
            // STEP 5: ALLOCATE OUTPUT MATRIX
            // ============================================================
            result.dims[0] = 1;
            result.dims[1] = 1;
            result.dims[2] = total_rows;
            result.dims[3] = total_cols;

            size_t total_elements = static_cast<size_t>(total_rows) * total_cols;
            result.data = std::make_unique<float[]>(total_elements);
            std::fill_n(result.data.get(), total_elements, 0.0f);

            // ============================================================
            // STEP 6: COLUMN START OFFSETS
            // ============================================================
            std::vector<int> col_starts(grid_cols + 1, 0);
            for (int i = 0; i < grid_cols; ++i) {
                col_starts[i + 1] = col_starts[i] + col_dims[i];
            }

            // ============================================================
            // STEP 7: PLACE SHARDS USING SHARD NUMBER (NOT LOOP INDEX)
            // ============================================================
            for (const auto& [shard_num, shard_bytes] : sorted_shards) {

                int row_idx = shard_num / grid_cols;   // A shard (0 or 1)
                int col_idx = shard_num % grid_cols;   // B shard

                const uint8_t* p2 = shard_bytes->data();
                p2 += sizeof(int); // ndim

                std::vector<int> dims(ndim);
                for (int i = 0; i < ndim; ++i) {
                    dims[i] = *reinterpret_cast<const int*>(p2);
                    p2 += sizeof(int);
                }

                int shard_rows_i = dims[2];
                int shard_cols_i = dims[3];
                const float* shard_data = reinterpret_cast<const float*>(p2);

                int dest_row = row_idx * shard_rows;
                int dest_col = col_starts[col_idx];

                std::cout << "DEBUG: Placing shard " << shard_num
                        << " → grid[" << row_idx << "][" << col_idx << "] "
                        << "(" << shard_rows_i << "×" << shard_cols_i << ")\n";

                for (int r = 0; r < shard_rows_i; ++r) {
                    std::memcpy(
                        result.data.get() + (dest_row + r) * total_cols + dest_col,
                        shard_data + r * shard_cols_i,
                        sizeof(float) * shard_cols_i
                    );
                }
            }

            std::cout << "DEBUG: System 2 grid assembly complete" << std::endl;
            return result;
        }

        MatrixResult combine_hierarchical_results(
            const combined_matrix_shards& combined,
            const std::vector<int>& split_order
        ) 
        {
            MatrixResult result;

            return result;
        }

        bool matrix_operation(
            const std::string& backend_type,
            const char* matrix_pathA,
            bool transposeA,
            const char* matrix_pathB,
            bool transposeB,
            bool use_gpu,
            int gpu_id,
            const std::string& send_back_str,  // CHANGED: Now a string for hierarchical info
            const std::string& operation_type,
            int dim,
            int shard_index_override,
            const std::vector<int>& hierarchical_split_order  // NEW parameter
        )
        {
            bool op_success = false;
            try {
                std::cout << "🚀 UNIFIED MATRIX OPERATION - Backend: " << backend_type << std::endl;

                // ============================================================
                // PARSE SEND_BACK INFORMATION
                // ============================================================
                int total_shards = 0;
                
                // Check if format contains '/' (new hierarchical format: "4/011" or "-4/011")
                size_t slash_pos = send_back_str.find('/');
                if (slash_pos != std::string::npos) {
                    // New format: "4/011" or "-4/011"
                    std::string total_shards_str = send_back_str.substr(0, slash_pos);
                    total_shards = std::stoi(total_shards_str);
                } else {
                    // Old format: just number
                    total_shards = std::stoi(send_back_str);
                }
                
                std::cout << "DEBUG: Parsed total_shards=" << total_shards 
                        << ", hierarchical_split_order size=" << hierarchical_split_order.size() << std::endl;

                // Common setup (all backends)
                std::string output_filename = get_matrix_output_filename(matrix_pathA, matrix_pathB);
                if (shard_index_override >= 0)
                {
                    // Force shard naming from caller
                    // Strip existing ".bin" and any trailing "_shard_<n>" before appending
                    size_t dot_pos = output_filename.rfind(".bin");
                    std::string base_name = (dot_pos != std::string::npos) ? output_filename.substr(0, dot_pos) : output_filename;
                    size_t shard_pos = base_name.rfind("_shard_");
                    if (shard_pos != std::string::npos)
                    {
                        base_name = base_name.substr(0, shard_pos);
                    }
                    output_filename = base_name + "_shard_" + std::to_string(shard_index_override) + ".bin";
                }
                std::string output_path = std::filesystem::path(matrix_shard_folder) / output_filename;

                // ============================================================
                // BACKEND: LLAMA / GGML / VULKAN
                // ============================================================
                if (backend_type == "llama")
                {
                    std::unique_ptr<float[]> matrix_A = nullptr;
                    std::unique_ptr<float[]> matrix_B = nullptr;
                    int rows_A, cols_A, rows_B, cols_B;
                    int depthA = 1, batchA = 1;
                    int depthB = 1, batchB = 1;

                    // Load matrices
                    matrix_A = load_matrix_bin(matrix_pathA, rows_A, cols_A, batchA, depthA);
                    matrix_B = load_matrix_bin(matrix_pathB, rows_B, cols_B, batchB, depthB);
                    
                    if (!matrix_A || !matrix_B) {
                        std::cerr << "❌ Failed to load input matrices" << std::endl;
                        return false;
                    }

                    // Apply transposes
                    if (transposeA) {
                        matrix_A = (depthA > 1 || batchA > 1)
                            ? matrix_backend_llama.transpose_4d(matrix_A.get(), batchA, depthA, rows_A, cols_A)
                            : matrix_backend_llama.transpose_2d(matrix_A.get(), rows_A, cols_A);
                        std::swap(rows_A, cols_A);
                    }
                    
                    if (transposeB) {
                        matrix_B = (depthB > 1 || batchB > 1)
                            ? matrix_backend_llama.transpose_4d(matrix_B.get(), batchB, depthB, rows_B, cols_B)
                            : matrix_backend_llama.transpose_2d(matrix_B.get(), rows_B, cols_B);
                        std::swap(rows_B, cols_B);
                    }

                    // GGML format: {cols, rows, depth, batch}
                    int dims_a[4] = { cols_A, rows_A, depthA, batchA };
                    int dims_b[4] = { cols_B, rows_B, depthB, batchB };

                    // Only lock long enough to read the backend vector; release before compute
                    ggml_backend_t backend;
                    {
                        std::lock_guard<std::mutex> lock(matrix_backend_llama.backends_mutex);
                        backend =
                            (use_gpu && gpu_id >= 0 &&
                            gpu_id < (int)matrix_backend_llama.ggml_backends.size())
                            ? matrix_backend_llama.ggml_backends[gpu_id]
                            : matrix_backend_llama.ggml_backends.back();
                    }

                    // Execute
                    MatrixResult result = matrix_backend_llama.matrix_op_nd(
                        matrix_A.get(), dims_a,
                        matrix_B.get(), dims_b,
                        backend, operation_type
                    );

                    if (result.dims[0] == 0 && result.dims[1] == 0) {
                        result.dims[0] = 1;
                        result.dims[1] = 1;
                    }

                    if (!result.data) {
                        std::cerr << "❌ LLAMA operation failed" << std::endl;
                        op_success = false;
                    } else {
                        // Common save/send_back
                        if (!save_matrix_bin(output_path.c_str(), result)) {
                            std::cerr << "❌ Failed to save result" << std::endl;
                            op_success = false;
                        } else {
                            if (total_shards != 0)  // Changed: check total_shards instead of send_back
                            {
                                // Pass hierarchical_split_order to send_back_file
                                send_back_file(output_path, output_filename, result, total_shards, 
                                            hierarchical_split_order, "llama");
                            }
                            
                            op_success = true;
                        }
                    }
                }

                // ============================================================
                // BACKEND: PYTORCH / TORCH
                // ============================================================
                else if (backend_type == "torch")
                {
                    // GPU availability check
                    bool torch_gpu_available = false;
                    #ifdef USE_CUDA
                    torch_gpu_available = torch::cuda::is_available();
                    #endif
                    
                    if (use_gpu && !torch_gpu_available) {
                        std::cout << "⚠️  GPU requested but unavailable. Using CPU." << std::endl;
                    }

                    // Load tensors
                    torch::Tensor A = load_matrix_bin_as_torch_view(matrix_pathA);
                    torch::Tensor B = load_matrix_bin_as_torch_view(matrix_pathB);
                    
                    if (!A.defined() || !B.defined()) {
                        std::cerr << "❌ Failed to load matrices" << std::endl;
                        op_success = false;
                        throw std::runtime_error("load fail");
                    }

                    // Apply transposes (last 2 dims)
                    if (transposeA)
                        A = A.transpose(-2, -1).contiguous();
                    if (transposeB)
                        B = B.transpose(-2, -1).contiguous();

                    // Select device
                    torch::Device device = torch::kCPU;
                    if (use_gpu && torch_gpu_available) {
                        device = torch::Device(torch::kCUDA, gpu_id);
                    }
                    A = A.to(device);
                    B = B.to(device);

                    // Execute
                    torch::Tensor C;
                    if (operation_type == "mul") {
                        C = torch::matmul(A, B);
                    } else if (operation_type == "add") {
                        C = A + B;
                    } else if (operation_type == "sub") {
                        C = A - B;
                    } else {
                        std::cerr << "❌ Unknown op: " << operation_type << std::endl;
                        return false;
                    }

                    C = C.contiguous().to(torch::kCPU);

                    // Convert to MatrixResult
                    MatrixResult result;
                    auto sizes = C.sizes();
                    int ndim = sizes.size();
                    int batch = 1, depth = 1, rows = 1, cols = 1;
                    
                    if (ndim == 2) {
                        rows = sizes[0]; cols = sizes[1];
                    } else if (ndim == 3) {
                        batch = sizes[0]; rows = sizes[1]; cols = sizes[2];
                    } else if (ndim == 4) {
                        batch = sizes[0]; depth = sizes[1]; rows = sizes[2]; cols = sizes[3];
                    } else {
                        std::cerr << "❌ Unsupported rank: " << ndim << std::endl;
                        return false;
                    }

                    result.dims[0] = batch;
                    result.dims[1] = depth;
                    result.dims[2] = rows;
                    result.dims[3] = cols;

                    int64_t total = C.numel();
                    result.data = std::make_unique<float[]>(total);
                    memcpy(result.data.get(), C.data_ptr<float>(), total * sizeof(float));

                    // Common save/send_back
                    if (!save_matrix_bin(output_path.c_str(), result)) {
                        std::cerr << "❌ Failed to save result" << std::endl;
                        op_success = false;
                    } else {
                        if (total_shards != 0)  // Changed: check total_shards instead of send_back
                        {
                            // Pass hierarchical_split_order to send_back_file
                            send_back_file(output_path, output_filename, result, total_shards, 
                                        hierarchical_split_order, "torch");
                        }
                        op_success = true;
                    }
                }

                // ============================================================
                // BACKEND: OPENCL
                // ============================================================
                else if (backend_type == "opencl")
                {
                    if (gpu_id < 0 || gpu_id >= (int)openCL_GPU_select_list.size()) {
                        std::cerr << "❌ Invalid OpenCL GPU ID" << std::endl;
                        return false;
                    }

                    // Load via Torch (I/O only)
                    torch::Tensor tensorA = load_matrix_bin_as_torch_view(matrix_pathA);
                    torch::Tensor tensorB = load_matrix_bin_as_torch_view(matrix_pathB);

                    if (!tensorA.defined() || !tensorB.defined()) {
                        std::cerr << "❌ Failed to load matrices" << std::endl;
                        return false;
                    }

                    // Apply transposes
                    if (transposeA)
                        tensorA = tensorA.transpose(-2, -1).contiguous();
                    if (transposeB)
                        tensorB = tensorB.transpose(-2, -1).contiguous();

                    float* A_ptr = tensorA.data_ptr<float>();
                    float* B_ptr = tensorB.data_ptr<float>();
                    int M = tensorA.size(-2);
                    int K = tensorA.size(-1);
                    int N = tensorB.size(-1);

                    if (K != tensorB.size(-2)) {
                        std::cerr << "❌ Dimension mismatch" << std::endl;
                        return false;
                    }

                    // OpenCL execution
                    cl::Device device = openCL_GPU_select_list[gpu_id];
                    cl::Context context(device);
                    cl::CommandQueue queue(context, device);
                    
                    cl::Buffer bufA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                sizeof(float) * tensorA.numel(), A_ptr);
                    cl::Buffer bufB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                sizeof(float) * tensorB.numel(), B_ptr);
                    cl::Buffer bufC(context, CL_MEM_WRITE_ONLY, sizeof(float) * M * N);
                    
                    cl::Program program(context, openCL_kernel_matmul);
                    program.build({device});
                    cl::Kernel kernel(program, "matmul");

                    kernel.setArg(0, bufA);
                    kernel.setArg(1, bufB);
                    kernel.setArg(2, bufC);
                    kernel.setArg(3, M);
                    kernel.setArg(4, N);
                    kernel.setArg(5, K);

                    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(M, N), cl::NDRange(16, 16));
                    queue.finish();

                    // Prepare result
                    MatrixResult result;
                    result.dims[0] = 1;
                    result.dims[1] = 1;
                    result.dims[2] = M;
                    result.dims[3] = N;
                    result.data = std::make_unique<float[]>(M * N);
                    queue.enqueueReadBuffer(bufC, CL_TRUE, 0, sizeof(float) * M * N, result.data.get());

                    // Common save/send_back
                    if (!save_matrix_bin(output_path.c_str(), result)) {
                        std::cerr << "❌ Failed to save result" << std::endl;
                        op_success = false;
                    } else {
                        if (total_shards != 0)  // Changed: check total_shards instead of send_back
                        {
                            // Pass hierarchical_split_order to send_back_file
                            send_back_file(output_path, output_filename, result, total_shards, 
                                        hierarchical_split_order, "opencl");
                        }
                        op_success = true;
                    }
                }

                else {
                    std::cerr << "❌ Unknown backend: " << backend_type << std::endl;
                    op_success = false;
                }
            }
            catch (const std::exception& e) {
                std::cerr << "❌ Exception: " << e.what() << std::endl;
                op_success = false;
            }

            try 
            {
                
                send_ack("ACK_matrixOp_complete"); 
            } 
            catch (...) 
            {}
            return op_success;
        }

        // OLD FUNCTIONS REMOVED - Now using unified matrix_operation() for all backends
        std::string get_matrix_output_filename(
            const std::string& matrix_pathA,
            const std::string& matrix_pathB
        )
        {
            // Extract filenames only
            std::string a_filename =
                std::filesystem::path(matrix_pathA).filename().string();
            std::string b_filename =
                std::filesystem::path(matrix_pathB).filename().string();

            // -------------------------------
            // Remove ".bin" if present
            // -------------------------------
            auto strip_bin = [](std::string& name)
            {
                size_t pos = name.rfind(".bin");
                if (pos != std::string::npos)
                    name = name.substr(0, pos);
            };

            strip_bin(a_filename);
            strip_bin(b_filename);

            // -------------------------------
            // Extract shard numbers
            // -------------------------------
            auto [matrix_nameA, shard_numA] =
                get_matrix_name_and_shard_number(a_filename);
            auto [matrix_nameB, shard_numB] =
                get_matrix_name_and_shard_number(b_filename);

            // -------------------------------
            // Remove any lingering "_shard_X"
            // -------------------------------
            auto strip_shard = [](std::string& name)
            {
                size_t pos = name.find("_shard_");
                if (pos != std::string::npos)
                    name = name.substr(0, pos);
            };

            strip_shard(matrix_nameA);
            strip_shard(matrix_nameB);

            // -------------------------------
            // Determine shard number
            // -------------------------------
            int shard_num = -1;
            if (shard_numA != -1)
                shard_num = shard_numA;
            else if (shard_numB != -1)
                shard_num = shard_numB;

            // -------------------------------
            // Build output filename
            // -------------------------------
            std::string output_filename =
                matrix_nameA + "x" + matrix_nameB;

            // If no shard number could be inferred, assign a sequential shard index per output base
            if (shard_num == -1) {
                std::lock_guard<std::mutex> lock(output_shard_mutex);
                shard_num = output_shard_counters[output_filename]++;
            }

            if (shard_num != -1)
                output_filename += "_shard_" + std::to_string(shard_num);

            output_filename += ".bin";

            return output_filename;
        }
};

int main()
{
    llama_zmq_server server;
    server.run_server();
}

/*
int main()
{
    const char* path_to_A = "/home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/test_2d_a.bin";
    const char* path_to_B = "/home/rino/Desktop/Open_Cluster_AI_Station_beta/cluster_matrix/matrix_shards/test_2d_a.bin";

    llama_zmq_server server;
    /*
    // ==========================================
    // Torch timing
    // ==========================================
    auto torch_start = std::chrono::high_resolution_clock::now();

    server.matrix_operation_torch(
        path_to_A,
        false,          // transposeA
        path_to_B,
        true,           // transposeB
        false,          // use_gpu
        0,              // gpu_id
        false,          // send_back
        "mul"
    );

    auto torch_end = std::chrono::high_resolution_clock::now();
    auto torch_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            torch_end - torch_start
        ).count();

    std::cout << "🔥 Torch took "
              << torch_us << " µs ("
              << torch_us / 1e6 << " s)\n";


    // ==========================================
    // GGML timing per backend
    // ==========================================
    for (int gpu_id = 0; gpu_id <= 2; gpu_id++) {

        auto start = std::chrono::high_resolution_clock::now();

        server.matrix_operation_llama(
            path_to_A,
            true,           // transposeA
            path_to_B,
            true,           // transposeB
            true,           // use_gpu
            gpu_id,         // gpu_id
            false,          // send_back
            "mul",
            2               // dim
        );

        auto end = std::chrono::high_resolution_clock::now();
        auto us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                end - start
            ).count();

        std::cout << "⚙️  GGML gpu_id " << gpu_id
                  << " took " << us << " µs ("
                  << us / 1e6 << " s)\n";
    }

}
*/

/*
int main()
{
    const char* path_to_B_shard_1 = "/dev/shm/matrix_shards/test_2d_a_shard_1.bin";
    const char* path_to_A        = "/dev/shm/matrix_shards/test_2d_a.bin";
    const char* path_to_B        = "/dev/shm/matrix_shards/test_2d_a.bin";

    llama_matrix_backend server;
    {
        std::unique_ptr<float[]> matrix_A = nullptr;
        std::unique_ptr<float[]> matrix_B = nullptr;
        std::unique_ptr<float[]> matrix_B_shard_1 = nullptr;

        int rows_A, cols_A;
        int rows_B, cols_B;
        int rows_B_shard_1, cols_B_shard_1;

        int depthA = 0, batchA = 0;
        int depthB = 0, batchB = 0;
        int depthB_s = 0, batchB_s = 0;

        matrix_A = load_matrix_bin(path_to_A, rows_A, cols_A, batchA, depthA);
        matrix_B = load_matrix_bin(path_to_B, rows_B, cols_B, batchB, depthB);
        matrix_B_shard_1 = load_matrix_bin(
            path_to_B_shard_1,
            rows_B_shard_1, cols_B_shard_1,
            batchB_s, depthB_s
        );

        // GGML dims: [cols, rows, depth, batch]
        int dims2d_a[4] = { cols_A, rows_A, 1, 1 };
        int dims2d_b[4] = { cols_B, rows_B, 1, 1 };
        int dims2d_b_shard_1[4] = { cols_B_shard_1, rows_B_shard_1, 1, 1 };

        std::cout << "Original A: " << rows_A << "x" << cols_A << std::endl;
        std::cout << "Original B (full): " << rows_B << "x" << cols_B << std::endl;
        std::cout << "Original B (shard_1): " << rows_B_shard_1 << "x" << cols_B_shard_1 << std::endl;

        //print_bin_from_torch("/dev/shm/matrix_shards/test_2d_a_shard_1.bin",10,10);
        //print_bin_from_torch("/dev/shm/matrix_shards/test_2d_a.bin",10,10);



        // ---- A @ B (shard_1) ----
        {
            MatrixResult r = server.matrix_op_nd(
                matrix_A.get(), dims2d_a,
                matrix_B_shard_1.get(), dims2d_b_shard_1,
                server.ggml_backends[0], "mul"
            );
            save_matrix_bin("/dev/shm/matrix_shards/shard_AB_out.bin",r);
            torch::Tensor shard_AB = load_matrix_bin_as_torch_view("/dev/shm/matrix_shards/shard_AB_out.bin");
            std::cout << "\nSHARD A@B flat:\n";
            print_tensor_start_flat(shard_AB, "fuck" ,40);
        }
        
    }

    return 0;
}
*/
