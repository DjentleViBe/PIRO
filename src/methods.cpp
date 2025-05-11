#include <vector>
#include <printutilities.hpp>
#include <extras.hpp>
#include <preprocess.hpp>
#include <logger.hpp>

namespace Piro::methods{
    float lookup(int row, int col, int N, std::vector<int>& Hash_keys_V, std::vector<float>& Hash_val_V, int TABLE_SIZE) {
        int index = row * N + col;
        int hash_index = index % TABLE_SIZE;
        int attempts = 0;
        int first_deleted = -1;
        // Linear probing to find the key
        while (Hash_keys_V[hash_index] != -1 && Hash_keys_V[hash_index] != index) {
            if (Hash_keys_V[hash_index] == -2 && first_deleted == -1) {
                first_deleted = hash_index;  // Remember the first deleted slot
            } 
            hash_index = (hash_index + 1) % TABLE_SIZE;
            attempts++;
            if (attempts >= TABLE_SIZE) {
                // logger::info("Error - lookhash [", hash_index, "]: Hash table [", TABLE_SIZE, "] is full. \n\t\t\t\t\t\t\tAttempts =", attempts,". \n\t\t\t\t\t\t\t");
                // Table is full, handle appropriately
                // std::exit(1);
                return 0.0; // Return error code
            }
        }
    
        if (Hash_keys_V[hash_index] == index) {
            // Key already exists → update value
            return Hash_val_V[hash_index];
        }
        return 0.0;
        // Key not found
    }
    
    int lookupandset(int row, int col, int N, float value, std::vector<int>& Hash_keys_V, std::vector<float>& Hash_val_V, int TABLE_SIZE) {
        int index = row * N + col;
        int hash_index = index % TABLE_SIZE;
        int attempts = 0;
        int first_deleted = -1;
        while(attempts <= TABLE_SIZE){
            if (Hash_keys_V[hash_index] == index){
                // Hash_val_V[hash_index] = value;
                return 0;
            }
            else if (Hash_keys_V[hash_index] == -1){
                int insert_index = (first_deleted != -1) ? first_deleted : hash_index;
                Hash_keys_V[insert_index] = index;
                Hash_val_V[insert_index] = value;
                return 0;
            }
            else if (Hash_keys_V[hash_index] == -2 && first_deleted == -1){
                first_deleted = hash_index;
            }
            hash_index = (hash_index + 1) % TABLE_SIZE;
            attempts++;
        }
        if(first_deleted != -1) {
            Hash_keys_V[first_deleted] = index;
            Hash_val_V[first_deleted] = value;
            return 0;
        }
        Piro::logger::info("Error - sethash [", hash_index, "/", index, "]: Hash table [", TABLE_SIZE, "] is full. \n\t\t\t\t\t\t\tAttempts =", attempts,", first_deleted = ", first_deleted,". Try reducing LoadFactor inside hashtable.ini. \n\t\t\t\t\t\t\tAborting program");
        Piro::logger::warning("hashkeys : ", Hash_keys_V);
        Piro::logger::warning("hashvalues : ", Hash_val_V);
        Piro::MeshParams& MP = Piro::MeshParams::getInstance();
        std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
        Piro::print_utilities::hash_to_dense_and_print(Hash_keys_V, Hash_val_V, n[0]*n[1]*n[2], TABLE_SIZE);
        std::exit(1);
        // Key not found
        return -1; // or some sentinel value for "not found"
    }
    
    int sethash(int index, float val, int TABLE_SIZE, std::vector<int>& Hash_keys_V, std::vector<float>& Hash_val_V) {
        int hash_index = index % TABLE_SIZE;
        int attempts = 0;
        int first_deleted = -1;
    
        while (attempts < TABLE_SIZE) {
            if (Hash_keys_V[hash_index] == index) {
                // Key found → update
                Hash_val_V[hash_index] = val;
                return 0;
            }
            else if (Hash_keys_V[hash_index] == -1) {
                // Empty slot → insert
                int insert_index = (first_deleted != -1) ? first_deleted : hash_index;
                Hash_keys_V[insert_index] = index;
                Hash_val_V[insert_index] = val;
                return 0;
            }
            else if (Hash_keys_V[hash_index] == -2 && first_deleted == -1) {
                // First deleted slot
                first_deleted = hash_index;
            }
    
            // Continue probing
            hash_index = (hash_index + 1) % TABLE_SIZE;
            attempts++;
        }
    
        // If full loop completed, insert at first_deleted if available
        if (first_deleted != -1) {
            Hash_keys_V[first_deleted] = index;
            Hash_val_V[first_deleted] = val;
            return 0;
        }
    
        // Table is truly full
        Piro::logger::info("Error - sethash [", hash_index, "/", index, "]: Hash table [", TABLE_SIZE, "] is full. \n\t\t\t\t\t\t\tAttempts =", attempts,", first_deleted = ", first_deleted,". Try reducing LoadFactor inside hashtable.ini. \n\t\t\t\t\t\t\tAborting program");
        Piro::logger::warning("hashkeys : ", Hash_keys_V);
        Piro::logger::warning("hashvalues : ", Hash_val_V);
        Piro::MeshParams& MP = Piro::MeshParams::getInstance();
        std::vector<uint> n = MP.getvalue<std::vector<uint>>(Piro::MeshParams::num_cells);
        Piro::print_utilities::hash_to_dense_and_print(Hash_keys_V, Hash_val_V, n[0]*n[1]*n[2], TABLE_SIZE);
        std::exit(1);
        return -1;
    }
}