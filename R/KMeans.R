
# DECLARED FUNCTION
normalize <- function(col){
  stdev = sd(col)
  if (stdev <= 0){
    col - mean(col)
  } else {
    (col - mean(col))/stdev
  }
}

# MAIN FLOW
rawData = csv.read("kddcup.data.corrected")

# All data are Non-NA.
all(!is.na(rawData))

# Print how many labels are there.
unique(sort(rawData[, ncol(rawData)]))

# Remove categorical columns.
data = rawData[, -c(2,3,4,ncol(rawData))]

# Data Normalization
# Shouldn't use these as columns with all identical values getting NaN after the scale().
# normalizedData = data.frame(scale(data))
normalizedData = data.frame(apply(data, 2, normalize))

# Don't know how ton configure epsilon like that in Spark!
# In Spark, controlling iter.max is no longer possible, but it is available in R!
model = kmeans(normalizedData, 5, iter.max = 10, nstart = 1)

# print the centroids
print(model$centers)

