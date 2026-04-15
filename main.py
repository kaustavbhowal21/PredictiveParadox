import pipeline as p

pipe = p.PipeLine('dataset/', True)
pipe.process()
pipe.predict('prediction.xlsx', regressor='LGBR')