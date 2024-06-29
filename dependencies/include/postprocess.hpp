#ifndef postprocess_hpp
#define postprocess_hpp

void postprocess();

template<typename T, typename... Args>
void postprocess(T first, Args... args);
void writevth();

#endif