# iterateve over all csv files in directory
date +%Y-%m-%d-%H:%M:%S
for f in *.csv
do
  echo "------------------------------------"
  # print file name
  echo "checking $f"
  # count commas in first line (header) and save to variable t
  ((T=$(head -n 1 $f | sed 's/[^,]//g' | wc -m) - 1 | bc -l))
  # print number of commas
  echo "$T commas found"
  C='{l=$0;t=gsub(/,/,"",l)}t=='$T
  C="awk '$C' $f > clean_$f"
  echo 'cleaning out bad lines:'
  echo $C
  eval $C
  date +%Y-%m-%d-%H:%M:%S
done
