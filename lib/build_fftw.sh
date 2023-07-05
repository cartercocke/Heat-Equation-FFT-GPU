# Define build paths
fftw_version="fftw-3.3.10"
fftw_tar="fftw-3.3.10.tar.gz"
fftw_source="https://www.fftw.org/$fftw_tar"
abs_path=$(pwd)
build_name="fftw-build"
build_path="$abs_path/$build_name"

if [ ! -d $build_name ] 
then
    # Download and extract fftw source code
    wget $fftw_source
    tar -xzf $fftw_tar
    rm $fftw_tar

    # Build fftw
    cd $fftw_version
    mkdir -p fftw-build
    ./configure --prefix=$build_path
    make
    make install
else
    echo "fftw already built"
fi
