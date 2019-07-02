import pdfx

# pdf = pdfx.PDFx("/Users/stephenbradshaw/Documents/codingTest/AutomaticKeyphraseExtraction-master/data/1/1.pdf")
# metadata = pdf.get_metadata()
#
# refs = pdf.get_references()
# ref_dict = pdf.get_references_as_dict()
#
#
# print(len(ref_dict))
# for k , v in ref_dict.items():
#     print(k ,  v)
# print(10*"=")
#
# metadata = pdf.get_metadata()
# print(metadata)
#
# print(10*"-=-")
# print(pdf.get_references_count())

from refextract import extract_references_from_file
reference = extract_references_from_file("/Users/stephenbradshaw/Documents/codingTest/AutomaticKeyphraseExtraction-master/data/1/1.pdf")
print(references)
