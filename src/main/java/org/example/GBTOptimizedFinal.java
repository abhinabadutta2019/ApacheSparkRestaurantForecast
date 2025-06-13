package org.example;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;

import org.apache.spark.ml.*;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.regression.GBTRegressor;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;

public class GBTOptimizedFinal {
    public static void main(String[] args) {

        SparkSession spark = SparkSession.builder()
                .appName("GBTOptimizedFinal")
                .master("local[*]")
                .config("spark.mongodb.read.connection.uri", "mongodb://127.0.0.1/db_spark.collection1")
                .getOrCreate();

        Dataset<Row> df = spark.read()
                .format("mongodb")
                .option("uri", "mongodb://127.0.0.1/db_spark.collection1")
                .load();

        // Drop missing and filter out high-revenue outliers
        Dataset<Row> filtered = df.na().drop()
                .filter("Revenue > 0 AND Revenue < 500000");

        // Add interaction features
        filtered = filtered
                .withColumn("Rating_x_Budget", functions.col("Rating").multiply(functions.col("Marketing Budget")))
                .withColumn("Ambience_x_Chef", functions.col("Ambience Score").multiply(functions.col("Chef Experience Years")));

        // Log transform on revenue
        filtered = filtered.withColumn("RevenueLog", functions.log1p(filtered.col("Revenue")));

        // Categorical and numerical features
        String[] catCols = {"Cuisine", "Location", "Parking Availability"};
        String[] numCols = {
                "Rating", "Marketing Budget", "Number of Reviews",
                "Average Meal Price", "Ambience Score", "Chef Experience Years",
                "Service Quality Score", "Social Media Followers",
                "Rating_x_Budget", "Ambience_x_Chef"
        };

        // Encoding categorical features
        String[] indexCols = new String[catCols.length];
        String[] oneHotCols = new String[catCols.length];
        StringIndexer[] indexers = new StringIndexer[catCols.length];
        OneHotEncoder[] encoders = new OneHotEncoder[catCols.length];

        for (int i = 0; i < catCols.length; i++) {
            indexCols[i] = catCols[i] + "_Index";
            oneHotCols[i] = catCols[i] + "_Vec";
            indexers[i] = new StringIndexer().setInputCol(catCols[i]).setOutputCol(indexCols[i]);
            encoders[i] = new OneHotEncoder().setInputCol(indexCols[i]).setOutputCol(oneHotCols[i]);
        }

        String[] allFeatures = new String[oneHotCols.length + numCols.length];
        System.arraycopy(oneHotCols, 0, allFeatures, 0, oneHotCols.length);
        System.arraycopy(numCols, 0, allFeatures, oneHotCols.length, numCols.length);

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(allFeatures)
                .setOutputCol("features");

        GBTRegressor gbt = new GBTRegressor()
                .setLabelCol("RevenueLog")
                .setFeaturesCol("features");

        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{
                indexers[0], indexers[1], indexers[2],
                encoders[0], encoders[1], encoders[2],
                assembler, gbt
        });

        // Train/test split
        Dataset<Row>[] splits = filtered.randomSplit(new double[]{0.8, 0.2}, 42);
        Dataset<Row> trainData = splits[0].cache();
        Dataset<Row> testData = splits[1];

        // Hyperparameter grid (fast)
        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(gbt.maxDepth(), new int[]{3})
                .addGrid(gbt.maxIter(), new int[]{40})
                .build();

        RegressionEvaluator logEval = new RegressionEvaluator()
                .setLabelCol("RevenueLog")
                .setPredictionCol("prediction")
                .setMetricName("rmse");

        TrainValidationSplit tvs = new TrainValidationSplit()
                .setEstimator(pipeline)
                .setEvaluator(logEval)
                .setEstimatorParamMaps(paramGrid)
                .setTrainRatio(0.8);

        // Start timing
        long start = System.currentTimeMillis();

        TrainValidationSplitModel model = tvs.fit(trainData);
        Dataset<Row> predictions = model.transform(testData);

        // End timing
        long end = System.currentTimeMillis();
        double duration = (end - start) / 1000.0;

        // Back-transform predictions to actual Revenue scale
        predictions = predictions.withColumn("prediction_actual", functions.expm1(predictions.col("prediction")));

        // Log-scale RMSE
        double rmseLog = logEval.evaluate(predictions);
        System.out.println("Log-scale RMSE: " + rmseLog);
        System.out.println("Time taken: " + duration + " seconds");

        // Actual RMSE (original Revenue scale)
        RegressionEvaluator realEval = new RegressionEvaluator()
                .setLabelCol("Revenue")
                .setPredictionCol("prediction_actual")
                .setMetricName("rmse");

        double realRmse = realEval.evaluate(predictions);
        System.out.println("Actual RMSE (Revenue scale): " + realRmse);

        // Export predictions
        predictions.select("Name", "Revenue", "prediction_actual")
                .coalesce(1)
                .write()
                .mode("overwrite")
                .option("header", "true")
                .csv("/home/abhinaba/Downloads/Codes/RestaurantForecast/output/gbt_final_log_fixed");

        spark.stop();
    }
}
