#ifndef methods_hpp
#define methods_hpp

#include <vector>

namespace Piro::methods{
    float lookup(int row, int col, int N, std::vector<int>& Hash_keys_V, std::vector<float>& Hash_val_V, int TABLE_SIZE);
    int lookupandset(int row, int col, int N, float value, std::vector<int>& Hash_keys_V, std::vector<float>& Hash_val_V, int TABLE_SIZE);
    int sethash(int index, float val, int TABLE_SIZE, std::vector<int>& Hash_keys_V, std::vector<float>& Hash_val_V);
    
}

#endif