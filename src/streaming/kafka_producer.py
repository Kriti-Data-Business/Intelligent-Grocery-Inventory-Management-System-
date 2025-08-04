from kafka import KafkaProducer, KafkaConsumer
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import time
import threading

class InventoryKafkaProducer:
    """Kafka producer for inventory data streaming"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.producer = None
        self.is_running = False
        
    def initialize_producer(self):
        """Initialize Kafka producer with error handling"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.config['bootstrap_servers'],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: str(k).encode('utf-8'),
                acks='all',
                retries=3,
                batch_size=16384,
                linger_ms=10,
                buffer_memory=33554432
            )
            logging.info("Kafka producer initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Kafka producer: {e}")
            raise
    
    def send_inventory_update(self, store_id: str, product_id: str, data: Dict):
        """Send inventory update to Kafka topic"""
        
        if not self.producer:
            self.initialize_producer()
        
        message = {
            'store_id': store_id,
            'product_id': product_id,
            'timestamp': datetime.now().isoformat(),
            **data
        }
        
        try:
            future = self.producer.send(
                self.config['inventory_topic'], 
                key=f"{store_id}:{product_id}",
                value=message
            )
            
            # Optional: wait for confirmation
            # record_metadata = future.get(timeout=10)
            # logging.debug(f"Message sent to {record_metadata.topic}:{record_metadata.partition}")
            
        except Exception as e:
            logging.error(f"Failed to send message: {e}")
    
    def simulate_real_time_data(self, duration_minutes: int = 60):
        """Simulate real-time inventory data for testing"""
        
        stores = ['store_001', 'store_002', 'store_003', 'store_004', 'store_005']
        products = ['prod_A', 'prod_B', 'prod_C', 'prod_D', 'prod_E']
        
        self.is_running = True
        start_time = datetime.now()
        
        logging.info(f"Starting data simulation for {duration_minutes} minutes")
        
        while self.is_running and (datetime.now() - start_time).total_seconds() < duration_minutes * 60:
            
            for store_id in stores:
                for product_id in products:
                    
                    # Generate realistic inventory data
                    current_hour = datetime.now().hour
                    base_demand = self._get_base_demand(product_id, current_hour)
                    
                    # Add some randomness
                    noise = np.random.normal(0, base_demand * 0.2)
                    current_inventory = max(0, np.random.poisson(base_demand * 5) + noise)
                    sales_last_hour = max(0, np.random.poisson(base_demand) + noise * 0.5)
                    
                    # Weather data
                    temperature = np.random.normal(25, 5)  # Celsius
                    humidity = np.random.normal(60, 15)    # Percentage
                    
                    # Business context
                    is_weekend = datetime.now().weekday() in [5, 6]
                    is_holiday = np.random.random() < 0.05  # 5% chance
                    promotion_active = np.random.random() < 0.1  # 10% chance
                    competitor_price = np.random.normal(100, 10)
                    
                    data = {
                        'current_inventory': float(current_inventory),
                        'sales_last_hour': float(sales_last_hour),
                        'temperature': float(temperature),
                        'humidity': float(humidity),
                        'is_weekend': bool(is_weekend),
                        'is_holiday': bool(is_holiday),
                        'competitor_price': float(competitor_price),
                        'promotion_active': bool(promotion_active)
                    }
                    
                    self.send_inventory_update(store_id, product_id, data)
            
            # Wait before next batch
            time.sleep(30)  # Send updates every 30 seconds
        
        logging.info("Data simulation completed")
    
    def _get_base_demand(self, product_id: str, hour: int) -> float:
        """Get base demand based on product and time"""
        
        # Product-specific demand patterns
        product_demand = {
            'prod_A': 10,  # Low demand product
            'prod_B': 25,  # Medium demand product  
            'prod_C': 50,  # High demand product
            'prod_D': 15,  # Low-medium demand
            'prod_E': 35   # Medium-high demand
        }
        
        base = product_demand.get(product_id, 20)
        
        # Time-based multipliers
        if 7 <= hour <= 9:    # Morning rush
            multiplier = 1.5
        elif 12 <= hour <= 14: # Lunch time
            multiplier = 1.8
        elif 17 <= hour <= 19: # Evening rush
            multiplier = 2.0
        elif 20 <= hour <= 22: # Dinner time
            multiplier = 1.3
        else:
            multiplier = 0.7
        
        return base * multiplier
    
    def stop_simulation(self):
        """Stop data simulation"""
        self.is_running = False
        if self.producer:
            self.producer.flush()
            self.producer.close()
        logging.info("Kafka producer stopped")

class InventoryKafkaConsumer:
    """Kafka consumer for processing inventory data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.consumer = None
        self.is_running = False
        self.message_handlers = []
    
    def initialize_consumer(self):
        """Initialize Kafka consumer"""
        try:
            self.consumer = KafkaConsumer(
                self.config['inventory_topic'],
                bootstrap_servers=self.config['bootstrap_servers'],
                group_id=self.config['consumer_group'],
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                auto_offset_reset='latest',
                enable_auto_commit=True,
                auto_commit_interval_ms=1000
            )
            logging.info("Kafka consumer initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Kafka consumer: {e}")
            raise
    
    def add_message_handler(self, handler_func):
        """Add message processing handler"""
        self.message_handlers.append(handler_func)
    
    def start_consuming(self):
        """Start consuming messages"""
        
        if not self.consumer:
            self.initialize_consumer()
        
        self.is_running = True
        logging.info("Starting message consumption")
        
        try:
            for message in self.consumer:
                if not self.is_running:
                    break
                
                # Process message with all handlers
                for handler in self.message_handlers:
                    try:
                        handler(message.key, message.value)
                    except Exception as e:
                        logging.error(f"Handler error: {e}")
                
        except KeyboardInterrupt:
            logging.info("Consumer interrupted by user")
        except Exception as e:
            logging.error(f"Consumer error: {e}")
        finally:
            self.stop_consuming()
    
    def stop_consuming(self):
        """Stop consuming messages"""
        self.is_running = False
        if self.consumer:
            self.consumer.close()
        logging.info("Kafka consumer stopped")
