from prepare.SyllableAnalyser import SyllableAnalyser
from prepare.SyllableSlicer import SyllableSlicer

words = ['check', 'bikes', 'sound', 'illegal', 'tiger', 'connection', 'unwritten', 'ruddy', 'aboriginal', 'alcoholic',
         'bait', 'apparel', 'wound', 'hole', 'babies', 'concentrate', 'punishment', 'sail', 'satisfy', 'cent',
         'abrasive', 'secretary', 'mailbox', 'juggle']

slicer = SyllableSlicer()
analyser = SyllableAnalyser()

slicer_result = map(lambda x: slicer.slice(x), words)
analyser_result = map(lambda x: analyser.slice(x), words)

for i, w in enumerate(words):
    print("%s:\t\t\t%s\t\t%s" % (w, '-'.join(slicer_result[i]), '-'.join(analyser_result[i])))
