/*******************************************************************************
 * Copyright (c) 2008-2009 The Khronos Group Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 ******************************************************************************/

/* $Revision: 10424 $ on $Date: 2010-02-18 04:04:49 +0530 (Thu, 18 Feb 2010) $ */

#ifndef __OPENCL_H
#define __OPENCL_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __APPLE__

#include <OpenCL/cl.h>
#include <OpenCL/cl_gl.h>
#include <OpenCL/cl_gl_ext.h>
#include <OpenCL/cl_ext.h>

#else

#include "cl.h"
#include "cl_gl.h"
#include "cl_gl_ext.h"
#include "cl_ext.h"

#endif

#ifdef __cplusplus
}
#endif

#endif  /* __OPENCL_H   */

