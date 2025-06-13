package org.example;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.evaluation.RegressionEvaluator;

public class MongoSparkLoader {
    public static void main(String[] args) {

        // 1. Create Spark session
        SparkSession spark = SparkSession.builder()
                .appName("MongoDBSparkRead")
                .master("local[*]")
                .config("spark.mongodb.read.connection.uri", "mongodb://127.0.0.1/db_spark.collection1")
                .getOrCreate();

        // 2. Load data from MongoDB
        Dataset<Row> df = spark.read()
                .format("mongodb")
                .option("uri", "mongodb://127.0.0.1/db_spark.collection1")
                .load();

        df.show(5); // Preview first 5 rows

        // 3. Prepare Data with 'Name' and ML columns
        Dataset<Row> dfWithName = df.select(
                "Name", "Rating", "Marketing Budget", "Number of Reviews",
                "Average Meal Price", "Ambience Score", "Chef Experience Years", "Revenue");

        // 4. Assemble features
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[] {
                        "Rating", "Marketing Budget", "Number of Reviews",
                        "Average Meal Price", "Ambience Score", "Chef Experience Years"
                })
                .setOutputCol("features");

        Dataset<Row> assembledData = assembler.transform(dfWithName)
                .select("Name", "features", "Revenue")
                .na().drop(); // remove rows with nulls

        // 5. Train Linear Regression model
        LinearRegression lr = new LinearRegression()
                .setLabelCol("Revenue")
                .setFeaturesCol("features");

        LinearRegressionModel model = lr.fit(assembledData);
        Dataset<Row> predictions = model.transform(assembledData);

        // 6. Evaluate model
        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol("Revenue")
                .setPredictionCol("prediction")
                .setMetricName("rmse");

        double rmse = evaluator.evaluate(predictions);
        System.out.println("Root Mean Squared Error (RMSE): " + rmse);

        // 7. Show and save predictions
        predictions.select("Name", "features", "Revenue", "prediction").show(5);

//        predictions.select("Name", "Revenue", "prediction")
//                .coalesce(1)
//                .write()
//                .option("header", "true")
//                .csv("/home/abhinaba/Downloads/Codes/RestaurantForecast/output/restaurant_predictions");
        predictions.select("Name", "Revenue", "prediction")
                .coalesce(1)
                .write()
                .mode("overwrite")  // <--- Add this line
                .option("header", "true")
                .csv("/home/abhinaba/Downloads/Codes/RestaurantForecast/output/restaurant_predictions");


        // 8. Stop Spark
        spark.stop();
    }
}
