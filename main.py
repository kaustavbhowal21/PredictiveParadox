import pipeline as p

pipe = p.PipeLine1(['dataset/train_demand_data.xlsx', 'dataset/train_weather_data.xlsx'], verbose=True)
pipe.train_model()
pipe.upload(['dataset/test_demand_data.xlsx', 'dataset/test_weather_data.xlsx'])
pipe.predict('prediction1.xlsx')

pipe = p.PipeLine2(['dataset/PGCB_date_power_demand.xlsx', 'dataset/weather_data.xlsx'], verbose=True)
pipe.split(2024)
pipe.train_model()
pipe.predict('prediction2.xlsx')