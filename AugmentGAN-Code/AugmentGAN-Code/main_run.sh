#!/bin/bash
#Our custom function
data=$1
aug_data=$5

#$2: Method either POS or Threshold
#$3 : It is number of folds u want
#$4 : Label number i.e, in imdb = 1 and in toxic : 6
#$5 : Location where augmented data stored
mkdir $aug_data
declare -i start=$3
declare -i end=${4}
cust_func1(){
	
  echo "Doing split $1 ..."

python3 augment_mine-Copy.py $2 $1 $3 $4 $5
  sleep 1
}


cust_func(){
	
  echo "Doing split $1 ..."
python3 augment_mine-Copy_pos.py $2 $1 $3 $4 $5
  sleep 1
}
echo "Step 1: Augmentation Process split dataset into 10 parts "
#Step 1 : Augmentation Process split dataset into 10 parts 
# For loop 10 times

if [ "$2" == 'pos' ]; then
for i in {1..5}
do
	cust_func $i $data $start $end $aug_data & # Put a function in the background
done
wait
fi

if [ "$2" == 'threshold' ]; then
for i in {1..5}
do
	cust_func1 $i $data $start $end $aug_data & # Put a function in the background
done
wait 
fi


## Put all cust_func in the background and bash 
## would wait until those are completed 
## before displaying all done message


echo "Step 2: Combining all the parts of dataset."
#Step2 : Combining all the parts of dataset.
if [ "$2" == 'pos' ]; then
	python3 combine.py aug_data
echo "Augmentation Done !!!"
fi
if [ "$2" == 'threshold' ]; then
	python3 combine.py aug_data
echo "Augmentation Done !!!"
fi