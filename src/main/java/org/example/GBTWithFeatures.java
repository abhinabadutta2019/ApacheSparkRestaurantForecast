package org.example;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import org.apache.spark.ml.*;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.regression.GBTRegressor;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;

public class GBTWithFeatures {
    public static void main(String[] args) {

        SparkSession spark = SparkSession.builder()
                .appName("GBTModelWithTuning")
                .master("local[*]")
                .config("spark.mongodb.read.connection.uri", "mongodb://127.0.0.1/db_spark.collection1")
                .getOrCreate();

        Dataset<Row> df = spark.read()
                .format("mongodb")
                .option("uri", "mongodb://127.0.0.1/db_spark.collection1")
                .load();

        // Feature columns
        String[] catCols = {"Cuisine", "Location", "Parking Availability"};
        String[] numCols = {
                "Rating", "Marketing Budget", "Number of Reviews",
                "Average Meal Price", "Ambience Score", "Chef Experience Years",
                "Service Quality Score", "Social Media Followers"
        };

        String[] indexCols = new String[catCols.length];
        String[] oneHotCols = new String[catCols.length];
        for (int i = 0; i < catCols.length; i++) {
            indexCols[i] = catCols[i] + "_Index";
            oneHotCols[i] = catCols[i] + "_Vec";
        }

        StringIndexer[] indexers = new StringIndexer[catCols.length];
        OneHotEncoder[] encoders = new OneHotEncoder[catCols.length];
        for (int i = 0; i < catCols.length; i++) {
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
                .setLabelCol("Revenue")
                .setFeaturesCol("features");

        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{
                indexers[0], indexers[1], indexers[2],
                encoders[0], encoders[1], encoders[2],
                assembler, gbt
        });

        Dataset<Row> cleanDF = df.selectExpr(
                "Name", "Cuisine", "Location", "`Parking Availability`",
                "Rating", "`Marketing Budget`", "`Number of Reviews`",
                "`Average Meal Price`", "`Ambience Score`", "`Chef Experience Years`",
                "`Service Quality Score`", "`Social Media Followers`", "Revenue"
        ).na().drop().filter("Revenue > 0 AND Revenue < 2000000");

        Dataset<Row>[] splits = cleanDF.randomSplit(new double[]{0.8, 0.2}, 42);
        Dataset<Row> trainData = splits[0];
        Dataset<Row> testData = splits[1];

        // ✅ Optional: Subsample training data for speed
        trainData = trainData.sample(0.5);  // 50% of training data

        // ✅ Small param grid for speed (2 combos only)
        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(gbt.maxDepth(), new int[]{3})
                .addGrid(gbt.maxIter(), new int[]{20, 40})
                .build();

        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol("Revenue")
                .setPredictionCol("prediction")
                .setMetricName("rmse");

        // ✅ Use TrainValidationSplit instead of CrossValidator
        TrainValidationSplit tvs = new TrainValidationSplit()
                .setEstimator(pipeline)
                .setEvaluator(evaluator)
                .setEstimatorParamMaps(paramGrid)
                .setTrainRatio(0.8);

        // ⏱️ Start timer
        long startTime = System.currentTimeMillis();

        TrainValidationSplitModel model = tvs.fit(trainData);
        Dataset<Row> predictions = model.transform(testData);

        // ⏱️ End timer
        long endTime = System.currentTimeMillis();
        double elapsedSecs = (endTime - startTime) / 1000.0;

        // 📉 Evaluate and save
        double rmse = evaluator.evaluate(predictions);
        System.out.println("Tuned GBT RMSE (test set): " + rmse);
        System.out.println("Time taken (seconds): " + elapsedSecs);

        predictions.select("Name", "Revenue", "prediction")
                .coalesce(1)
                .write()
                .mode("overwrite")
                .option("header", "true")
                .csv("/home/abhinaba/Downloads/Codes/RestaurantForecast/output/gbt_tuned_predictions");

        spark.stop();
    }
}
