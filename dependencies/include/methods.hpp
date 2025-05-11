#ifndef methods_hpp
#define methods_hpp

#include <vector>

namespace Piro::methods{
    /**
     * @brief Hash Table lookup function.
     *
     * This function returns value given the row and column of the matrix.
     *
     * @param row row number. 
     * @param col column number.
     * @param N number of rows in teh matrix.
     * @param Hash_keys_V hash keys.
     * @param Hash_val_V hashvalues.
     * @param TABLE_SIZE hash table size
     * @return returns the look up value, 0 if not found.
     */
    float lookup(int row, int col, int N, std::vector<int>& Hash_keys_V, std::vector<float>& Hash_val_V, int TABLE_SIZE);
    /**
     * @brief Hash Table lookup and set function.
     *
     * This function returns value given the row and column of the matrix.
     *
     * @param row row number. 
     * @param col column number.
     * @param N Total number of columns.
     * @param value value to be set.
     * @param Hash_keys_V hash keys.
     * @param Hash_val_V hashvalues.
     * @param TABLE_SIZE hash table size
     * @return returns -1 if not found.
     */
    int lookupandset(int row, int col, int N, float value, std::vector<int>& Hash_keys_V, std::vector<float>& Hash_val_V, int TABLE_SIZE);
    /**
     * @brief Hash Table set function.
     *
     * This function returns value given the row and column of the matrix.
     *
     * @param index index value. 
     * @param val value to to set.
     * @param TABLE_SIZE table size.
     * @param Hash_keys_V hash keys.
     * @param Hash_val_V hash values.
     * @return returns -1 if not found.
     */
    int sethash(int index, float val, int TABLE_SIZE, std::vector<int>& Hash_keys_V, std::vector<float>& Hash_val_V);
    
}

#endif