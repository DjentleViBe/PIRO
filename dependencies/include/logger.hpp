#ifndef logger_hpp
#define logger_hpp

#include <vector>
#include <iostream>
#include <printutilities.hpp>

extern int debuginfo;

namespace Piro{
    class logger {
        public:
            template<typename... Args>
            static void info(const Args&... args) {
                if(debuginfo >= 0){
                    Piro::print_utilities::print_time();
                    ((std::cout << args << " "), ...);
                    std::cout << "" << std::endl;
                }
            }
            
            template<typename... Args>
            static void error(const Args&... args) {
                if(debuginfo >= 3){
                    Piro::print_utilities::print_time();
                    ((std::cout << args << " "), ...);
                    std::cout << "" << std::endl;
                }
            }

            template<typename... Args>
            static void debug(const Args&... args) {
                if(debuginfo >= 1){
                    Piro::print_utilities::print_time();
                    ((std::cout << args << " "), ...);
                    std::cout << "" << std::endl;
                }
            }
            template<typename... Args>
            static void debug_print(const Args&... args) {
                if(debuginfo >= 1){
                    ((std::cout << args << " "), ...);
                    // std::cout << "" << std::endl;
                }
            }

            // Basic case with a single argument
            template<typename T>
            static void info_debug(const T& arg) {
                std::cout << arg;
            }
            
            // Specialization for vectors
            template<typename T>
            static void info_debug(const std::vector<T>& vec) {
                for (const auto& item : vec) {
                    std::cout << item << " ";
                }
            }

            // Variadic template version for multiple arguments
            template<typename T, typename... Args>
            static void warning(const T& first, const Args&... args) {
                if(debuginfo >= 2){
                    Piro::print_utilities::print_time();
                    info_debug(first);  // Handle the first argument
                    info_debug(args...); // Recursive call for remaining arguments
                    std::cout << "" << std::endl;
                }
                
            }
        };
}
#endif