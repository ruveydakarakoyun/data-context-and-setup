import pandas as pd
import numpy as np
from olist.utils import haversine_distance
from olist.data import Olist


class Order:
    '''
    DataFrames containing all orders as index,
    and various properties of these orders as columns
    '''
    def __init__(self):
        # Assign an attribute ".data" to all new instances of Order
        self.data = Olist().get_data()

    def get_wait_time(self, is_delivered=True):
        orders = self.data["orders"].copy()

        if is_delivered:
            orders = orders[
                orders["order_status"] == "delivered"
            ].copy()

        date_columns = [
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]

        orders[date_columns] = orders[date_columns].apply(
            pd.to_datetime
        )

        orders["wait_time"] = (
            orders["order_delivered_customer_date"]
            - orders["order_purchase_timestamp"]
        ).dt.total_seconds() / (24 * 60 * 60)

        orders["expected_wait_time"] = (
            orders["order_estimated_delivery_date"]
            - orders["order_purchase_timestamp"]
        ).dt.total_seconds() / (24 * 60 * 60)

        orders["delay_vs_expected"] = (
            orders["order_delivered_customer_date"]
            - orders["order_estimated_delivery_date"]
        ).dt.total_seconds() / (24 * 60 * 60)

        orders["delay_vs_expected"] = (
            orders["delay_vs_expected"].clip(lower=0)
        )

        return orders[
            [
                "order_id",
                "wait_time",
                "expected_wait_time",
                "delay_vs_expected",
                "order_status",
            ]
        ]

    def get_review_score(self):
        reviews = self.data["order_reviews"].copy()

        reviews["dim_is_five_star"] = (
            reviews["review_score"] == 5
        ).astype(int)

        reviews["dim_is_one_star"] = (
            reviews["review_score"] == 1
        ).astype(int)

        return reviews[
            [
                "order_id",
                "dim_is_five_star",
                "dim_is_one_star",
                "review_score",
            ]
        ]

    def get_number_items(self):
        order_items = self.data["order_items"].copy()

        number_items = (
            order_items
            .groupby("order_id", as_index=False)
            .agg(number_of_items=("order_item_id", "count"))
        )

        return number_items

    def get_number_sellers(self):
        order_items = self.data["order_items"].copy()

        number_sellers = (
            order_items
            .groupby("order_id", as_index=False)
            .agg(number_of_sellers=("seller_id", "nunique"))
        )

        return number_sellers

    def get_price_and_freight(self):
        order_items = self.data["order_items"].copy()

        price_and_freight = (
            order_items
            .groupby("order_id", as_index=False)
            .agg(
                price=("price", "sum"),
                freight_value=("freight_value", "sum")
            )
        )

        return price_and_freight

    # Optional
    def get_distance_seller_customer(self):

        # Her posta kodu için ortalama koordinat
        geolocation = (
            self.data["geolocation"]
            .groupby(
                "geolocation_zip_code_prefix",
                as_index=False,
            )
            .agg(
                geolocation_lat=("geolocation_lat", "mean"),
                geolocation_lng=("geolocation_lng", "mean"),
            )
        )

        # Müşteri koordinatları
        customers_with_geo = (
            self.data["customers"][
                [
                    "customer_id",
                    "customer_zip_code_prefix",
                ]
            ]
            .merge(
                geolocation,
                left_on="customer_zip_code_prefix",
                right_on="geolocation_zip_code_prefix",
                how="left",
            )
            .rename(
                columns={
                    "geolocation_lat": "customer_lat",
                    "geolocation_lng": "customer_lng",
                }
            )
        )

        # Satıcı koordinatları
        sellers_with_geo = (
            self.data["sellers"][
                [
                    "seller_id",
                    "seller_zip_code_prefix",
                ]
            ]
            .merge(
                geolocation,
                left_on="seller_zip_code_prefix",
                right_on="geolocation_zip_code_prefix",
                how="left",
            )
            .rename(
                columns={
                    "geolocation_lat": "seller_lat",
                    "geolocation_lng": "seller_lng",
                }
            )
        )

        # Aynı siparişte aynı satıcıyı bir kez tut
        order_sellers = (
            self.data["order_items"][
                [
                    "order_id",
                    "seller_id",
                ]
            ]
            .drop_duplicates()
        )

        # Sipariş, müşteri ve satıcı koordinatlarını birleştir
        matching_geo = (
            self.data["orders"][
                [
                    "order_id",
                    "customer_id",
                ]
            ]
            .merge(
                customers_with_geo[
                    [
                        "customer_id",
                        "customer_lat",
                        "customer_lng",
                    ]
                ],
                on="customer_id",
                how="left",
            )
            .merge(
                order_sellers,
                on="order_id",
                how="inner",
            )
            .merge(
                sellers_with_geo[
                    [
                        "seller_id",
                        "seller_lat",
                        "seller_lng",
                    ]
                ],
                on="seller_id",
                how="left",
            )
            .dropna(
                subset=[
                    "customer_lat",
                    "customer_lng",
                    "seller_lat",
                    "seller_lng",
                ]
            )
            .copy()
        )

        # Haversine mesafesi, kilometre cinsinden
        matching_geo["distance_seller_customer"] = matching_geo.apply(
            lambda row: haversine_distance(
                row["customer_lng"],
                row["customer_lat"],
                row["seller_lng"],
                row["seller_lat"],
            ),
            axis=1,
        )

        # Birden fazla satıcılı siparişlerde ortalama mesafe
        distance_seller_customer = (
            matching_geo
            .groupby(
                "order_id",
                as_index=False,
            )
            .agg(
                distance_seller_customer=(
                    "distance_seller_customer",
                    "mean",
                )
            )
        )

        return distance_seller_customer

    def get_training_data(
        self,
        is_delivered=True,
        with_distance_seller_customer=False
    ):
        training_data = (
            self.get_wait_time(is_delivered=is_delivered)
            .merge(
                self.get_review_score(),
                on="order_id",
                how="inner"
            )
            .merge(
                self.get_number_items(),
                on="order_id",
                how="inner"
            )
            .merge(
                self.get_number_sellers(),
                on="order_id",
                how="inner"
            )
            .merge(
                self.get_price_and_freight(),
                on="order_id",
                how="inner"
            )
        )

        if with_distance_seller_customer:
            training_data = training_data.merge(
                self.get_distance_seller_customer(),
                on="order_id",
                how="inner"
            )

        return training_data.dropna()
