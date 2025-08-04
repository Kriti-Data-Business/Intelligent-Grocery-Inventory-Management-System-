from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
import logging
from typing import Dict, List

class SparkInventoryProcessor:
    """Real-time inventory processing using Spark Streaming"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.spark = None
        self.feature_columns = config.get('feature_columns', [])
        self.model = None
        
    def initialize_spark(self):
        """Initialize Spark session with optimized configuration"""
        
        self.spark = SparkSession.builder \
            .appName(self.config['spark']['app_name']) \
            .config("spark.executor.memory", self.config['spark']['executor_memory']) \
            .config("spark.driver.memory", self.config['spark']['driver_memory']) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.streaming.checkpointLocation", "/tmp/checkpoint") \
            .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        logging.info("Spark session initialized successfully")
    
    def create_streaming_pipeline(self, kafka_config: Dict):
        """Create end-to-end streaming pipeline"""
        
        if not self.spark:
            self.initialize_spark()
        
        # Read from Kafka stream
        raw_stream = self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", kafka_config['bootstrap_servers']) \
            .option("subscribe", kafka_config['input_topic']) \
            .option("startingOffsets", "latest") \
            .option("failOnDataLoss", "false") \
            .load()
        
        # Parse JSON data
        parsed_stream = raw_stream.select(
            col("key").cast("string"),
            from_json(col("value").cast("string"), self._get_input_schema()).alias("data"),
            col("timestamp")
        ).select("key", "data.*", "timestamp")
        
        # Feature engineering
        featured_stream = self._apply_feature_engineering(parsed_stream)
        
        # Real-time predictions
        prediction_stream = self._apply_predictions(featured_stream)
        
        # Inventory optimization
        optimized_stream = self._apply_optimization(prediction_stream)
        
        # Write results to output topic
        query = optimized_stream.select(
            col("store_id").cast("string").alias("key"),
            to_json(struct("*")).alias("value")
        ).writeStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", kafka_config['bootstrap_servers']) \
            .option("topic", kafka_config['output_topic']) \
            .option("checkpointLocation", "/tmp/checkpoint/output") \
            .outputMode("append") \
            .trigger(processingTime='30 seconds') \
            .start()
        
        return query
    
    def _get_input_schema(self):
        """Define input data schema"""
        return StructType([
            StructField("store_id", StringType(), True),
            StructField("product_id", StringType(), True),
            StructField("timestamp", TimestampType(), True),
            StructField("current_inventory", DoubleType(), True),
            StructField("sales_last_hour", DoubleType(), True),
            StructField("temperature", DoubleType(), True),
            StructField("humidity", DoubleType(), True),
            StructField("is_weekend", BooleanType(), True),
            StructField("is_holiday", BooleanType(), True),
            StructField("competitor_price", DoubleType(), True),
            StructField("promotion_active", BooleanType(), True)
        ])
    
    def _apply_feature_engineering(self, df):
        """Apply real-time feature engineering"""
        
        # Temporal features
        df_featured = df.withColumn("hour", hour(col("timestamp"))) \
                       .withColumn("day_of_week", dayofweek(col("timestamp"))) \
                       .withColumn("month", month(col("timestamp")))
        
        # Cyclical encoding
        df_featured = df_featured \
            .withColumn("hour_sin", sin(2 * 3.14159 * col("hour") / 24)) \
            .withColumn("hour_cos", cos(2 * 3.14159 * col("hour") / 24)) \
            .withColumn("day_sin", sin(2 * 3.14159 * col("day_of_week") / 7)) \
            .withColumn("day_cos", cos(2 * 3.14159 * col("day_of_week") / 7))
        
        # Weather interaction features
        df_featured = df_featured \
            .withColumn("temp_humidity_interaction", col("temperature") * col("humidity") / 100)
        
        # Business logic features
        df_featured = df_featured \
            .withColumn("inventory_turnover_rate", 
                       when(col("current_inventory") > 0, 
                            col("sales_last_hour") / col("current_inventory")).otherwise(0))
        
        # Rolling aggregations using window functions
        window_spec = Window.partitionBy("store_id", "product_id") \
                           .orderBy(col("timestamp").cast("long")) \
                           .rangeBetween(-3600, 0)  # 1 hour window
        
        df_featured = df_featured \
            .withColumn("avg_sales_1h", avg("sales_last_hour").over(window_spec)) \
            .withColumn("max_sales_1h", max("sales_last_hour").over(window_spec))
        
        return df_featured
    
    def _apply_predictions(self, df):
        """Apply ML predictions in streaming context"""
        
        # Prepare features for ML model
        feature_cols = [
            "hour_sin", "hour_cos", "day_sin", "day_cos",
            "temperature", "humidity", "temp_humidity_interaction",
            "current_inventory", "inventory_turnover_rate",
            "avg_sales_1h", "max_sales_1h"
        ]
        
        # Vector assembler
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        df_vector = assembler.transform(df)
        
        # Load pre-trained model (in production, this would be loaded from model registry)
        if not self.model:
            self.model = self._load_streaming_model()
        
        # Make predictions
        predictions = self.model.transform(df_vector)
        
        # Add prediction confidence intervals
        predictions = predictions \
            .withColumn("demand_prediction_lower", col("prediction") * 0.8) \
            .withColumn("demand_prediction_upper", col("prediction") * 1.2)
        
        return predictions
    
    def _apply_optimization(self, df):
        """Apply inventory optimization logic"""
        
        # Simple reorder point logic for real-time processing
        df_optimized = df \
            .withColumn("reorder_point", 
                       col("prediction") * 1.5 + 10) \
            .withColumn("max_stock_level", 
                       col("prediction") * 3 + 50) \
            .withColumn("recommended_order", 
                       when(col("current_inventory") < col("reorder_point"),
                            col("max_stock_level") - col("current_inventory"))
                       .otherwise(0)) \
            .withColumn("urgency_level",
                       when(col("current_inventory") < col("prediction"), "HIGH")
                       .when(col("current_inventory") < col("reorder_point"), "MEDIUM")
                       .otherwise("LOW"))
        
        return df_optimized
    
    def _load_streaming_model(self):
        """Load pre-trained model for streaming predictions"""
        
        # In production, load from MLflow or similar model registry
        # For demo purposes, create a simple GBT model
        gbt = GBTRegressor(
            featuresCol="features",
            labelCol="prediction",  # This would be trained offline
            predictionCol="prediction",
            maxIter=10
        )
        
        # This would be a pre-trained model loaded from storage
        # For now, return the untrained model object
        logging.info("Streaming model loaded (placeholder)")
        return gbt
    
    def start_monitoring_dashboard(self):
        """Start real-time monitoring dashboard"""
        
        # Create monitoring queries
        monitoring_query = self.spark.sql("""
            CREATE OR REPLACE TEMPORARY VIEW inventory_metrics AS
            SELECT 
                store_id,
                COUNT(*) as total_products,
                AVG(current_inventory) as avg_inventory,
                AVG(prediction) as avg_predicted_demand,
                SUM(CASE WHEN urgency_level = 'HIGH' THEN 1 ELSE 0 END) as high_urgency_count,
                MAX(timestamp) as last_update
            FROM streaming_inventory_data
            GROUP BY store_id
        """)
        
        return monitoring_query
