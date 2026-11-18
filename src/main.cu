//---------------------------------------------------------------------
// Standard Libraries
//---------------------------------------------------------------------
#include <iostream>  
#include <random> 

//---------------------------------------------------------------------
// Graph data-structure
//---------------------------------------------------------------------
#include "graph/graph.cuh" 

//---------------------------------------------------------------------
// Utility Functions
//---------------------------------------------------------------------
#include "utility/timer.hpp"             
#include "utility/utility.hpp"           
#include "utility/cuda_utility.cuh"
#include "utility/CommandLineParser.cuh" 

//---------------------------------------------------------------------
// Biconnected Components (BCC) related functions
//---------------------------------------------------------------------
#include "extern_bcc/extern_bcc.cuh"
#include "extern_bcc/bcg_memory_utils.cuh"
#include "extern_bcc/serial_bcc.cuh"


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool    checker             = false;
bool    g_verbose           = false;  // Whether to display i/o to console
bool    g_gpu_bcc           = false;  // -g: dense ids + GPU Compress-Graph
int     local_block_size    = 1000;
long    maxThreadsPerBlock  = 0;

//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::ios_base::sync_with_stdio(false);

    // Initialize command line parser and retrieve parsed arguments.
    CommandLineParser cmdParser(argc, argv);
    const auto& args = cmdParser.getArgs();
        
    if (args.error) {
        std::cerr << CommandLineParser::help_msg << std::endl;
        exit(EXIT_FAILURE);
    }

    try {
        std::string filename = args.inputFile;
        std::cout <<"\nReading " << get_file_extension(filename) << " file." << std::endl << std::endl;
        undirected_graph g(filename);

        int     num_vert    =   g.getNumVertices();
        long    num_edges   =   g.getNumEdges() / 2;
        
        // Initialize device
        cuda_init(args.cudaDevice);
        g_verbose = args.verbose;
        g_gpu_bcc = args.gpu_bcc;
        checker   = args.g_checker_mode;

        // print basic_stats
        g.basic_stats(maxThreadsPerBlock, g_verbose, checker);

        if(g_verbose) {
            std::cout <<"\n\nEdgeList:\n";
            g.print_edgelist();
        }
        // Create External - BCC object
        GPU_BCG gpu_bcg(num_vert, num_edges, args.batchSize);

        gpu_bcg.h_edgelist = g.getList();

        std::cout << "Starting External BCC:" << std::endl;
        
        // start external_bcc
        int bcc_count = extern_bcc(gpu_bcg);

        std::cout << "The number of biconnected components in " << get_file_extension(filename) << " is: " << bcc_count << "\n";

        // -c : ground truth. One serial Tarjan pass over the whole edge stream.
        // This is the reference, not the product -- it reads every edge at
        // once, which is exactly the cost the single-pass algorithm exists to
        // avoid. It is here so a run can be checked, not benchmarked.
        if (checker) {
            Timer serial_timer;
            long bridges = 0, isolated = 0;
            long exact = serial_bcc_count(g.getList(), num_edges, num_vert,
                                          &bridges, &isolated);
            double serial_ms = serial_timer.stop();

            std::cout << "\n========================================\n"
                      << "       Serial reference (exact)\n"
                      << "========================================\n\n"
                      << "Biconnected components:   " << exact    << "\n"
                      << "  of which bridges:       " << bridges  << "\n"
                      << "  2-connected blocks:     " << (exact - bridges) << "\n"
                      << "Isolated vertices:        " << isolated
                      << "  (not counted above)\n"
                      << "Serial Tarjan time:       " << serial_ms << " ms\n"
                      << "========================================\n";
        }

    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

// ====[ End of Main Code ]====