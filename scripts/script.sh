mkdir "gridcorpus"
cd "gridcorpus"
mkdir "raw" "video" "words"
cd "raw" && mkdir "video" "words"

for i in `seq $1 $2`
do
    printf "\n\n------------------------- Downloading $i-th video and words alignment -------------------------\n\n"
    cd "words" && curl "http://spandh.dcs.shef.ac.uk/gridcorpus/s$i/align/s$i.tar" > "s$i.tar" && cd ..
    #cd "video" && curl "http://spandh.dcs.shef.ac.uk/gridcorpus/s$i/video/s$i.mpg_vcd.zip" > "s$i.zip" && cd ..
    #unzip -q "video/s$i.zip" -d "../video" && rm "video/s$i.zip"
    mkdir -p "../words/s$i" && tar -xf "words/s$i.tar" -C "../words/s$i" && rm "words/s$i.tar"
done
