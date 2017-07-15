package com.mycompany

import java.nio.file.{Files, Paths}

import org.apache.commons.lang3.exception.ExceptionUtils
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD

/**
 * @author ${user.name}
 */
object App {

  // Used to compute the distance between a point and another point (typically a centroid).
  def distance(a: Vector, b: Vector): Double = {
    math.sqrt(a.toArray.zip(b.toArray).map({case (x1, x2) => x1 - x2}).map(x => x * x).sum)
  }

  // Given a vector, predict if that point belongs to a certain centroid.
  def distanceToCentroid(a: Vector, model: KMeansModel) = {
    val cluster = model.predict(a)
    val centroid = model.clusterCenters(cluster)
    distance(a, centroid)
  }

  def normalize(a: Vector, stdevs: Array[Double], means: Array[Double]): Vector = {
    val res = (a.toArray, stdevs, means).zipped.map({case (x, stdev, mean) =>
      if (stdev <= 0) (x - mean) else (x - mean)/stdev})
    Vectors.dense(res)
  }

  def clusteringScore(data: RDD[Vector], k: Int): Double ={
    val kmeans = new KMeans()
    kmeans.setK(k)
    val model = kmeans.run(data)
    data.map(a => distanceToCentroid(a, model)).mean()
  }

  def clusteringScoreWithEntropy(normalizedLabelsAndData: RDD[(String, Vector)], k: Int): Double = {

    val kmeans = new KMeans()
    kmeans.setK(k)
    val model = kmeans.run(normalizedLabelsAndData.values)
    val LabelsAndClusters = normalizedLabelsAndData.mapValues(x => model.predict(x))
    val ClustersAndLabels = LabelsAndClusters.map(_.swap)
    val n = normalizedLabelsAndData.count()
    val ClustersAndTheirLabelCounts = ClustersAndLabels.groupByKey().mapValues(x => x.groupBy(l => l).map(_._2.size))

    ClustersAndTheirLabelCounts.values.map(labelCount => labelCount.sum * entropy(labelCount)).sum() / n
  }

  def entropy(counts: Iterable[Int]): Double = {
    val values = counts.filter(x => x > 0) // The labels that aren't there in the cluster shouldn't be considered.
    val sum = values.sum
    values.map(x => {
      val proportion = x.toDouble / sum
      -proportion * math.log(proportion)
    }).sum
  }

  def main(args : Array[String]) {

    println("*** START ***")

    Logger.getLogger("org").setLevel(Level.OFF);
    Logger.getLogger("akka").setLevel(Level.OFF);

    val DATA_PATH = "./mydata/kddcup.data.corrected"

    try{

      val conf = new SparkConf().setMaster("local[*]").setAppName("Max Price")
      val sc = new SparkContext(conf)

      if (!Files.exists(Paths.get(DATA_PATH))){
        throw new Exception("[%s] not found.".format(DATA_PATH))
      }

      //////////////////////////////////////////////////////////////////////////////////////////////
      // DATA PROCESSING
      val rawData = sc.textFile(DATA_PATH)

      // Print how many labels are there.
      rawData.map(_.split(",").last).countByValue().toSeq.sortBy(_._2).reverse.foreach(println)

      // K-Means can't play with categorical variables. So remove them.
      val labelsAndData = rawData.map(line => {
        val buffer = line.split(",").toBuffer
        buffer.remove(1, 3)
        val label = buffer.remove(buffer.length - 1)
        val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
        (label, vector) // We will need to labels later!
      }).cache()

      val data: RDD[Vector] = labelsAndData.values

      // Data Normalization
      // Calculate Standard Deviation & Muy
      val dataAsArray = data.map(_.toArray)
      val numCols = dataAsArray.first().length
      val sums = dataAsArray.reduce({case (arrA, arrB) => arrA.zip(arrB).map({case (a, b) => a + b})})
      val sumSquares = dataAsArray.aggregate(new Array[Double](numCols))({case (accuArr, xArr) => accuArr.zip(xArr).map({case (accu, x) => accu + x*x})},
                                {case (accu1Arr, accu2Arr) => accu1Arr.zip(accu2Arr).map({case (accu1, accu2) => accu1 + accu2})})
      val n = dataAsArray.count()
      // https://www.sciencebuddies.org/science-fair-projects/project_data_analysis_variance_std_deviation.shtml
      // Look at "There's a more efficient way"
      val stdevs = sumSquares.zip(sums).map({case (sq, s) => math.sqrt(n*sq - s*s)/n})
      val means = sums.map(_ / n)

      // Normalize the data
      val normalizedData = data.map(vec => normalize(vec, stdevs, means)).cache()

      //////////////////////////////////////////////////////////////////////////////////////////////
      // K-MEANS WITH DEFAULT K
      // org.apache.spark.mllib.clustering.KMeans
      // For RDD.
      // If wanting Dataframe, Data Pipeline... ==> org.apache.spark.ml.clustering.KMeans
      val kmeans = new KMeans()
      // Default k = 2
      val model = kmeans.run(normalizedData)
      // Let's also control the number of runs and epsilons.
      //kmeans.setRuns(10)
      // See more:
      // No effect since Spark 2.0 (https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/mllib/clustering/KMeans.scala)
      // https://issues.apache.org/jira/browse/SPARK-11358
      // https://issues.apache.org/jira/browse/SPARK-11560
      kmeans.setEpsilon(1.0e-6)

      // Print cluster centers.
      println("Cluster Centers:")
      model.clusterCenters.foreach(println)

      val clusterLabelCount = labelsAndData.map({ case (label, vector) =>
        val prediction = model.predict(vector)
        (prediction, label)
      }).countByValue()

      clusterLabelCount.toArray.foreach({case ((cluster, label), count) =>
        // http://docs.scala-lang.org/overviews/core/string-interpolation.html
        // Use prefix "f" to make it "printf".
        println(f"$cluster%1s$label%18s$count%8d")
      })

      //////////////////////////////////////////////////////////////////////////////////////////////
      // K-MEANS WITH DIFFERENT Ks WITH DISTANCE AS SCORING
      // Let's try some k
      println("\nTesting with some k (with Distance as ClusteringScore):")
      (60 to 120 by 10).map(k => {
        println(f"\nCurrent: $k")
        (k, clusteringScore(normalizedData, k))}).toList.sortBy(_._2).foreach(println)

      //////////////////////////////////////////////////////////////////////////////////////////////
      // K-MEANS WITH DIFFERENT Ks WITH ENTROPY AS SCORING
      println("\nTesting with some k (with Entropy as ClusteringScore):")
      val LabelsAndNormalizedData = labelsAndData.map({case (label, vector) => (label, normalize(vector, stdevs, means))}).cache()

      (60 to 120 by 10).map(k => {
        println(f"\nCurrent: $k")
        (k, clusteringScoreWithEntropy(LabelsAndNormalizedData, k))
      }).toList.sortBy(_._2).foreach(println)

      //////////////////////////////////////////////////////////////////////////////////////////////
      // CALCULATE THE THRESHOLD: 100th-farthest data point from among known data.
      val threshold = normalizedData.map(vec => {
        distanceToCentroid(vec, model)
      }).top(100).last

      // For any new data point, use distanceToCentroid() to compute its distance to its closest centroid.
      // If the distance is larger than the threshold above ==> worth checking!

    }
    catch {
      case e: Throwable => println(ExceptionUtils.getStackFrames(e))
    }

    println("*** END ***")
  }
}
