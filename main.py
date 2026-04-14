import pipeline as p

pipe = p.PipeLine('dataset/', False)
pipe.process()
pipe.predict('prediction.xlsx')