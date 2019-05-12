
# Other models
* Gaussian Naive-Bayes (GaussianNB) - ???
* K-nearest neighbors - Classifier implementing the k-nearest neighbors vote
* Gaussian process classification (GPC) - based on Laplace approximation. Internally, the Laplace approximation is used for approximating the non-Gaussian posterior by a Gaussian. Currently, the implementation is restricted to using the logistic link function.
* Neural Net (MLP) - Multi-layer Perceptron classifier. This model optimizes the log-loss function using LBFGS or stochastic gradient descent.

# Procedure


* One-hot encoded categorical variables
* Split data into features and labels
* Converted to arrays
* Split data into training and testing sets

# Findings
* Many visits to the website are short
* very few `sessionsummary` entries probably due to client side privacy tools such as tracking blockers.



# Dataset
## Integrity due user behavior
In the month of august

## Size
The size of the dataset is an issue. The dataset for just the month of august is 10GB in size.

I couldn't find any information on the encoding of CSV files and considering Pandas' `read_csv` can't decode them out of the box neither with **UTF-8** nor **ISO-8859-1**, I've resorted to removing bad lines.

For each csv file in a directory:
* read number of commas in first line
* write lines from file with same number of commas to a new file
* new file gets named `clean_{filename.ext}`

So `page_aug.csv` becomes `clean_page_aug.csv`.

```bash
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
```
