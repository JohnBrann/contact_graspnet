# CUDA_INCLUDE=' -I/usr/local/cuda/include/'
# CUDA_LIB=' -L/usr/local/cuda/lib64/'
# TF_CFLAGS=$(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
# TF_LFLAGS=$(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
# cd pointnet2/tf_ops/sampling

# nvcc -std=c++11 -c -o tf_sampling_g.cu.o tf_sampling_g.cu \
#  ${CUDA_INCLUDE} ${TF_CFLAGS} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# g++ -std=c++11 -shared -o tf_sampling_so.so tf_sampling.cpp \
#  tf_sampling_g.cu.o ${CUDA_INCLUDE} ${TF_CFLAGS} -fPIC -lcudart ${TF_LFLAGS} ${CUDA_LIB}

# echo 'testing sampling'
# python3 tf_sampling.py
 
# cd ../grouping

# nvcc -std=c++11 -c -o tf_grouping_g.cu.o tf_grouping_g.cu \
#  ${CUDA_INCLUDE} ${TF_CFLAGS} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# g++ -std=c++11 -shared -o tf_grouping_so.so tf_grouping.cpp \
#  tf_grouping_g.cu.o ${CUDA_INCLUDE} ${TF_CFLAGS} -fPIC -lcudart ${TF_LFLAGS} ${CUDA_LIB}

# echo 'testing grouping'
# python3 tf_grouping_op_test.py

 
# cd ../3d_interpolation
# g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -shared -fPIC ${TF_CFLAGS} ${TF_LFLAGS} -O2
# echo 'testing interpolate'
# python3 tf_interpolate_op_test.py

#!/usr/bin/env bash
set -e

# Gather TensorFlow compile and link flags
tf_cflags=$(python3 - <<EOF
import tensorflow as tf
print(" ".join(tf.sysconfig.get_compile_flags()))
EOF
)
tf_lflags=$(python3 - <<EOF
import tensorflow as tf
print(" ".join(tf.sysconfig.get_link_flags()))
EOF
)

# Gather TensorFlow include directories (including nsync)
tf_inc=$(python3 - <<EOF
import tensorflow as tf, os
inc = tf.sysconfig.get_include()
nsync = os.path.join(inc, 'external', 'nsync', 'public')
paths = [inc]
if os.path.isdir(nsync): paths.append(nsync)
print(":".join(paths))
EOF
)
IFS=":" read -r -a TF_INC_ARRAY <<< "$tf_inc"
TF_INC_FLAGS=""
for d in "${TF_INC_ARRAY[@]}"; do
  TF_INC_FLAGS+=" -I$d"
done

# CUDA include and library paths
CUDA_INCLUDE="-I/usr/local/cuda/include"
CUDA_LIB="-L/usr/local/cuda/lib64 -lcudart"

echo "Starting compilation of PointNet TF ops"

build_op() {
  local dir="$1"; local op="$2"
  pushd "$dir" > /dev/null
  echo "Building $op in $dir"

  # Compile CUDA kernel
  nvcc -std=c++11 -c -o ${op}_g.cu.o ${op}_g.cu \
    ${CUDA_INCLUDE} ${TF_INC_FLAGS} ${tf_cflags} -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

  # Link shared object with TensorFlow and CUDA
  g++ -std=c++11 -shared -fPIC -o ${op}_so.so ${op}.cpp ${op}_g.cu.o \
    ${CUDA_INCLUDE} ${CUDA_LIB} ${TF_INC_FLAGS} ${tf_cflags} ${tf_lflags}

  echo "Testing $op"
  python3 ${op}.py
  popd > /dev/null
}

# Build sampling and grouping ops
build_op "pointnet2/tf_ops/sampling" "tf_sampling"
build_op "pointnet2/tf_ops/grouping" "tf_grouping"

# Build 3d_interpolation op
echo "Building tf_interpolate"
pushd pointnet2/tf_ops/3d_interpolation > /dev/null
  # Compile and link 3d interpolation op
  g++ -std=c++11 -shared -fPIC -o tf_interpolate_so.so tf_interpolate.cpp \
    ${CUDA_INCLUDE} ${CUDA_LIB} ${TF_INC_FLAGS} ${tf_cflags} ${tf_lflags}
  echo "Testing interpolate"
  python3 tf_interpolate_op_test.py
popd > /dev/null

echo "All PointNet TF ops built successfully."

