#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "../dependencies/include/Python.h"
#include "../dependencies/include/python_performance.hpp"
#include "../dependencies/include/arrayobject.h"
#include "../dependencies/include/preprocess.hpp"
#include "../dependencies/include/extras.hpp"
#include <vector>
#include <iostream>

#define import_array_check(){ if (_import_array() < 0) { PyErr_Print();PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");return -1; }}
int ia(){
    import_array_check();
    return 0;
}

std::vector<float> mul_using_numpy(std::vector<std::vector<float>> A, std::vector<float> B){
    // Py_Initialize();
    // ia();
    size_t num_rows = A.size();
    size_t num_cols = (num_rows > 0) ? A[0].size() : 0;
    size_t total_elements = num_rows * num_cols;
    
    // Flatten the 2D vector into a 1D vector
    std::vector<float> flattened;
    flattened.reserve(total_elements);
    for (const auto& row : A) {
        flattened.insert(flattened.end(), row.begin(), row.end());
    }
    
    // import_array_check();
    // Define array dimensions (2D array with 3x3 shape)
    // Define dimensions for the matrices
    npy_intp dims1[2] = {MP.n[0]*MP.n[1]*MP.n[2], MP.n[0]*MP.n[1]*MP.n[2]};  // Matrix A: 2x3
    npy_intp dims2[2] = {MP.n[0]*MP.n[1]*MP.n[2], 1};  // Matrix B: 3x2
    
    // Create NumPy arrays
    PyObject *array1 = PyArray_SimpleNew(2, dims1, NPY_FLOAT);
    PyObject *array2 = PyArray_SimpleNew(2, dims2, NPY_FLOAT);
    
    // Fill array1 and array2 with data (for demonstration)
     // Fill array1 with data from flattened (Matrix A)
    float *data1 = (float *)PyArray_DATA((PyArrayObject *)array1);
    for (int i = 0; i < MP.n[0] * MP.n[1] * MP.n[2] * MP.n[0] * MP.n[1] * MP.n[2]; i++) {
        data1[i] = flattened[i];
    }

    // Fill array2 with data from B (Matrix B)
    float *data2 = (float *)PyArray_DATA((PyArrayObject *)array2);
    for (int i = 0; i < MP.n[0] * MP.n[1] * MP.n[2]; i++) {
        data2[i] = B[i];
    }
    std::cout << "matmulstarted" << std::endl;
    print_time();
    // Perform matrix multiplication (A * B)
    PyObject *pModule = PyImport_ImportModule("numpy");
    PyObject *pFunc = PyObject_GetAttrString(pModule, "dot");
    PyObject *pArgs = PyTuple_Pack(2, array1, array2);
    PyObject *result = PyObject_CallObject(pFunc, pArgs);
    std::cout << "matmulend" << std::endl;
    print_time();
    
    // Retrieve result into a std::vector<float>
    std::vector<float> R;
    if (result && PyArray_Check(result)) {
        float *data = (float *)PyArray_DATA((PyArrayObject *)result);
        R.assign(data, data + PyArray_SIZE((PyArrayObject *)result));
    } else {
        std::cerr << "Error: Failed to perform matrix multiplication.\n";
    }
    
    // Clean up
    Py_DECREF(array1);
    Py_DECREF(array2);
    Py_DECREF(result);
    Py_DECREF(pArgs);
    Py_DECREF(pFunc);
    Py_DECREF(pModule);

    // Py_Finalize();

    return R;

}

int callpython(){
    // Py_Initialize must be called before any other 
    // Python C/API functions
    Py_Initialize();
    import_array_check();
    // Equivalent of "import platform" in Python
    PyObject * platform_module = PyImport_ImportModule("platform");
    if(!platform_module) {
        // The import failed.
        // Double check spelling in import call and confirm
        // you can import platform in Python on system.
        printf("\nCould not import platform module.\n");
        return 0;
    }

    PyObject* system_function = PyObject_GetAttrString(platform_module, "system");
    if(system_function && PyCallable_Check(system_function)) {

        // Call platform.system()
        PyObject * system_obj = PyObject_CallObject(system_function, NULL);

        // Convert PyObject * returned by above call into a human-readable string
        // we can print to stdin
        PyObject* str = PyUnicode_AsEncodedString(system_obj, "utf-8", "~E~");
        const char * system_str = PyBytes_AS_STRING(str);
        printf("\n\nPlatform name is %s \n\n", system_str);

        // Decrement reference counts for objects we are 
        // done using to allow garbage collection to dispose of them
        Py_XDECREF(system_function);
        Py_XDECREF(system_obj);
        Py_XDECREF(str);
    } else {
        // Something has gone wrong.
        // Double check spelling in PyObject_GetAttrString and try
        // calling platform.system() in Python.
        printf("\nCould not call platform.system() function.\n");
    }     

    // Finalize the Python Interpreter
    
    // Py_Finalize();
    
    return 0;
}

