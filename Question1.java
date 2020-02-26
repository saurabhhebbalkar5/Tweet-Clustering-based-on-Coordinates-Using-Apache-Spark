package saurabh_assignment4;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

/**
 *
 * @author Saurabh
 */
public class Question1 {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        
        //Spark Configuration
        SparkConf sparkConf = new SparkConf()
                .setAppName("WeatherStation")
                .setMaster("local[4]").set("spark.executor.memory", "1g");

        //Intializing spark context
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        //Importing file
        String path = "F:\\NUIG -AI\\3. Large Scale Data Analytics\\Assignment 4\\twitter2D.txt"; // here, the data is numerical already
        JavaRDD<String> data = sc.textFile(path);
        
        JavaPairRDD<String, Vector> parsedData = data.mapToPair(
                (String s) -> {
                    //Splitting the data by comma
                    String[] arrayData = s.split(",");
                    if (arrayData.length > 5) {
                        //few tweet text start at different position hence checking the array length and handling them
                        arrayData[4] = arrayData[4].concat(arrayData[5]);
                    }
                    //different array to specifically handle lat, long corodtinates
                    double[] coordinates = new double[2];
                    for (int i = 0; i < 2; i++) {
                        coordinates[i] = Double.parseDouble(arrayData[i]);
                    }
                    return new Tuple2<>(s, Vectors.dense(coordinates));
                }
        );
        System.out.println("Clustering Tweet based on coordinates");
        parsedData.cache();
        // Cluster the data into two classes using KMeans
        int numClusters = 4;
        int numIterations = 2000;

        KMeansModel clusters = KMeans.train(parsedData.values().rdd(), numClusters, numIterations);

        JavaPairRDD<Integer, String> result = parsedData.mapToPair(f -> {
            //splitting the data seperated by comma
            String[] array = f._1.split(",");
            if (array.length > 5) {
                //few tweet text start at different position hence checking the array length and handling them 
                array[4] = array[5];
            }
            //returning key value pair
            return new Tuple2<>(clusters.predict(f._2), array[4]);
        });
        //sorting the data based on cluster
        JavaPairRDD<Integer, String> result1 = result.sortByKey();
        
        //printing the tweet along with clusters based on cluster sort
        result1.sortByKey().collect().forEach((str) -> {
            System.out.println("Tweet: \"" + str._2 + "\" is in Cluster " + str._1);
        });

        sc.stop();
        sc.close();

    }

}
