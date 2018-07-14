Dear Sir or Madam,

Welcome to this small dataset related to Punctuation Restoration or Sentence Boundary Detection.

The data consist of the transcripts of TED talks and originally prepared in IWSLT 2011 evaluation campaign. I'm sorry that I cannot presicely recall how I preprocessed these data from IWSLT package, but currently it contains:

train2012 - the 2.1M words training set

dev2012 - the 296k words dev set

test2011 - the 12626 words test set for manually generated transcripts

test2011asr - the 12828 words test set for ASR output

IWSLT12.TED.MT.tst2011.en-fr.en.xml - the source file of test2011, fully punctuated

IWSLT12.TED.SLT.tst2011.en-fr.en.system0.comma.xml - the source file of test2011asr, I personally added the commas to the unpunctuated ASR output, so there may be some errors inside.

---------------------------------------------------------

There are only 3 types of punctuation marks in this dataset: comma, period and question mark.

When dealing with the text, I treat the phenomenon like "don't" or "you're" seperately into "do" + "n't", "you" + "'re", which may be different from other researchers. Feel free to adjust them into any format you think as the best.

And also there is an error in line 130775 in train2012, Jiangyan Yi (jiangyan.yi@nlpr.ia.ac.cn) pointed it out. Perhaps there are more errors, if you find it, please tell me, thank you.

----------------------------------------------------------

Until Sept. 2016, this dataset is used in following papers:

"Punctuation prediction for unsegmented transcript based on word vector"
by Xiaoyin Che, Cheng Wang, Haojin Yang and Christoph Meinel.

"Sentence Boundary Detection Based on Parallel Lexical and Acoustic Models"
by Xiaoyin Che, Sheng Luo, Haojin Yang and Christoph Meinel.

"Bidirectional Recurrent Neural Network with Attention Mechanism for Punctuation Restoration"
by Ottokar Tilk and Tanel Alumae

If you want to compare the result with some of the above papers, please also include them into reference.

----------------------------------------------------------

Please contact me by email: xiaoyin.che@hpi.de, if you have any further questions.

Hope you can get your desired result.

Xiaoyin Che