# create multi label lmdb

label.txt like this:
0000001.jpg 0
0000001.jpg 0 1 2 3
0000001.jpg 124 244 12 87

just support uint8 label

--help:
----------------------------------------
usage:    
--help    show help 
--shuffle shuffle image
-l        label.txt 
-i        image dir 
-o        output    
example:
python create_lmdb.py -l label.txt -i path/to/image/ -o train_lmdb/
----------------------------------------

