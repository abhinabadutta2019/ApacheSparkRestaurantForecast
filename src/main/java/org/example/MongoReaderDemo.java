package org.example;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class MongoReaderDemo {
    public static void main(String[] args) {

        // 1. Start Spark session
        SparkSession spark = SparkSession.builder()
                .appName("MongoDBReadTest")
                .master("local[*]")
                .config("spark.mongodb.read.connection.uri", "mongodb://127.0.0.1/db_spark.collection1")
                .getOrCreate();

        // 2. Read from MongoDB
        Dataset<Row> df = spark.read()
                .format("mongodb")
                .option("uri", "mongodb://127.0.0.1/db_spark.collection1")
                .load();

        // 3. Show first 10 rows
        df.show(10);

        // 4. Stop Spark
        spark.stop();
    }
}
