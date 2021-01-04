from flair.data import Sentence
from flair.models import SequenceTagger

s = Sentence('I love Berlin')
t = SequenceTagger.load('ner')

t.predict(s)

print(s)
for entity in s.get_spans('ner'):
    print(entity)
