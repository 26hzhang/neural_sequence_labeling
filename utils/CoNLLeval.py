import argparse
import json
import re
import sys
from collections import defaultdict

"""
Borrowed from: https://github.com/AdolfVonKleist/rnn-slu/blob/master/rnnslu/CoNLLeval.py
"""


class CoNLLeval:
    """Evaluate the result of processing CoNLL-2000 shared task
    Evaluate the result of processing CoNLL-2000 shared tasks. This is a
    vanilla python port of the original perl script.
    # usage:     conlleval [-l] [-r] [-d delimiterTag] [-o oTag] < file
    #            README: http://cnts.uia.ac.be/conll2000/chunking/output.html
    # options:   l: generate LaTeX output for tables like in
    #               http://cnts.uia.ac.be/conll2003/ner/example.tex
    #            r: accept raw result tags (without B- and I- prefix;
    #                                       assumes one word per chunk)
    #            d: alternative delimiter tag (default is single space)
    #            o: alternative outside tag (default is O)
    # note:      the file should contain lines with items separated
    #            by $delimiter characters (default space). The final
    #            two items should contain the correct tag and the
    #            guessed tag in that order. Sentences should be
    #            separated from each other by empty lines or lines
    #            with $boundary fields (default -X-).
    # url:       http://lcg-www.uia.ac.be/conll2000/chunking/
    """

    def __init__(self, verbose=0, raw=False, delimiter=" ", otag="O", boundary="-X-"):
        self.verbose = verbose  # verbosity level
        self.boundary = boundary  # sentence boundary
        self.correct = None  # current corpus chunk tag (I,O,B)
        self.correct_chunk = 0  # number of correctly identified chunks
        self.correct_tags = 0  # number of correct chunk tags
        self.correct_type = None  # type of current corpus chunk tag (NP,VP,etc.)
        self.delimiter = delimiter  # field delimiter
        self.FB1 = 0.0  # FB1 score (Van Rijsbergen 1979)
        self.accuracy = 0.0
        self.first_item = None  # first feature (for sentence boundary checks)
        self.found_correct = 0  # number of chunks in corpus
        self.found_guessed = 0  # number of identified chunks
        self.guessed = None  # current guessed chunk tag
        self.guessed_type = None  # type of current guessed chunk tag
        self.i = None  # miscellaneous counter
        self.in_correct = False  # currently processed chunk is correct until now
        self.last_correct = "O"  # previous chunk tag in corpus
        self.latex = 0  # generate LaTeX formatted output
        self.last_correct_type = ""  # type of previously identified chunk tag
        self.last_guessed = "O"  # previously identified chunk tag
        self.last_guessed_type = ""  # type of previous chunk tag in corpus
        self.last_type = None  # temporary storage for detecting duplicates
        self.line = None  # line
        self.nbr_of_features = -1  # number of features per line
        self.precision = 0.0  # precision score
        self.o_tag = otag  # outside tag, default O
        self.raw = raw  # raw input: add B to every token
        self.recall = 0.0  # recall score
        self.token_counter = 0  # token counter (ignores sentence breaks)

        self.correct_chunk = defaultdict(int)  # number of correctly identified chunks per type
        self.found_correct = defaultdict(int)  # number of chunks in corpus per type
        self.found_guessed = defaultdict(int)  # number of identified chunks per type

        self.features = []  # features on line
        self.sorted_types = []  # sorted list of chunk type names

    @staticmethod
    def endOfChunk(prev_tag, tag, prev_type, tag_type, chunk_end=0):
        """Checks if a chunk ended between the previous and current word.
        Checks if a chunk ended between the previous and current word.
        Args:
            prev_tag (str): Previous chunk tag identifier.
            tag (str): Current chunk tag identifier.
            prev_type (str): Previous chunk type identifier.
            tag_type (str): Current chunk type identifier.
            chunk_end (int): 0/True true/false identifier.
        Returns:
            int: 0/True true/false identifier.
        """
        if prev_tag == "B" and tag == "B":
            chunk_end = True
        if prev_tag == "B" and tag == "O":
            chunk_end = True
        if prev_tag == "I" and tag == "B":
            chunk_end = True
        if prev_tag == "I" and tag == "O":
            chunk_end = True
        if prev_tag == "E" and tag == "E":
            chunk_end = True
        if prev_tag == "E" and tag == "I":
            chunk_end = True
        if prev_tag == "E" and tag == "O":
            chunk_end = True
        if prev_tag == "I" and tag == "O":
            chunk_end = True
        if prev_tag != "O" and prev_tag != "." and prev_type != tag_type:
            chunk_end = True
        # corrected 1998-12-22: these chunks are assumed to have length 1
        if prev_tag == "]":
            chunk_end = True
        if prev_tag == "[":
            chunk_end = True

        return chunk_end

    @staticmethod
    def startOfChunk(prevTag, tag, prevType, tag_type, chunk_start=0):
        """Checks if a chunk started between the previous and current word.
        Checks if a chunk started between the previous and current word.
        Args:
            prevTag (str): Previous chunk tag identifier.
            tag (str): Current chunk tag identifier.
            prevType (str): Previous chunk type identifier.
            tag_type (str): Current chunk type identifier.
            chunk_start:
        Returns:
            int: 0/True true/false identifier.
        """
        if prevTag == "B" and tag == "B":
            chunk_start = True
        if prevTag == "I" and tag == "B":
            chunk_start = True
        if prevTag == "O" and tag == "B":
            chunk_start = True
        if prevTag == "O" and tag == "I":
            chunk_start = True
        if prevTag == "E" and tag == "E":
            chunk_start = True
        if prevTag == "E" and tag == "I":
            chunk_start = True
        if prevTag == "O" and tag == "E":
            chunk_start = True
        if prevTag == "O" and tag == "I":
            chunk_start = True
        if tag != "O" and tag != "." and prevType != tag_type:
            chunk_start = True
        # corrected 1998-12-22: these chunks are assumed to have length 1
        if tag == "[":
            chunk_start = True
        if tag == "]":
            chunk_start = True
        return chunk_start

    def Evaluate(self, infile):
        """Evaluate test outcome for a CoNLLeval shared task.
        Evaluate test outcome for a CoNLLeval shared task.
        Args:
            infile (str): The input file for evaluation.
        """
        with open(infile, "r") as ifp:
            for line in ifp:
                line = line.lstrip().rstrip()
                self.features = re.split(self.delimiter, line)
                if len(self.features) == 1 and re.match(r"^\s*$", self.features[0]):
                    self.features = []
                if self.nbr_of_features < 0:
                    self.nbr_of_features = len(self.features) - 1
                elif self.nbr_of_features != len(self.features) - 1 and len(self.features) != 0:
                    raise ValueError("Unexpected number of features: {0}\t{1}".format(len(self.features) + 1,
                                                                                      self.nbr_of_features + 1))
                if len(self.features) == 0 or self.features[0] == self.boundary:
                    self.features = [self.boundary, "O", "O"]
                if len(self.features) < 2:
                    raise ValueError("CoNLLeval: Unexpected number of features in line.")

                if self.raw is True:
                    if self.features[-1] == self.o_tag:
                        self.features[-1] = "O"
                    if self.features[-2] == self.o_tag:
                        self.features[-2] = "O"
                    if not self.features[-1] == "O":
                        self.features[-1] = "B-{0}".format(self.features[-1])
                    if not self.features[-2] == "O":
                        self.features[-2] = "B-{0}".format(self.features[-2])
                # 20040126 ET code which allows hyphens in the types
                ffeat = re.search(r"^([^\-]*)-(.*)$", self.features[-1])
                if ffeat:
                    self.guessed = ffeat.groups()[0]
                    self.guessed_type = ffeat.groups()[1]
                else:
                    self.guessed = self.features[-1]
                    self.guessed_type = ""

                self.features.pop(-1)
                ffeat = re.search(r"^([^\-]*)-(.*)$", self.features[-1])
                if ffeat:
                    self.correct = ffeat.groups()[0]
                    self.correct_type = ffeat.groups()[1]
                else:
                    self.correct = self.features[-1]
                    self.correct_type = ""
                self.features.pop(-1)

                if self.guessed_type is None:
                    self.guessed_type = ""
                if self.correct_type is None:
                    self.correct_type = ""

                self.first_item = self.features.pop(0)

                # 1999-06-26 sentence breaks should always be counted as out of chunk
                if self.first_item == self.boundary:
                    self.guessed = "O"

                if self.in_correct is True:
                    if self.endOfChunk(self.last_correct, self.correct, self.last_correct_type,
                                       self.correct_type) is True and self.endOfChunk(self.last_guessed, self.guessed,
                                                                                      self.last_guessed_type,
                                                                                      self.guessed_type) is True \
                            and self.last_guessed_type == self.last_correct_type:
                        self.in_correct = False
                        self.correct_chunk[self.last_correct_type] += 1
                    elif self.endOfChunk(self.last_correct, self.correct, self.last_correct_type,
                                         self.correct_type) != self.endOfChunk(self.last_guessed, self.guessed,
                                                                               self.last_guessed_type,
                                                                               self.guessed_type) \
                            or self.guessed_type != self.correct_type:
                        self.in_correct = False

                if self.startOfChunk(self.last_correct, self.correct, self.last_correct_type,
                                     self.correct_type) is True and self.startOfChunk(self.last_guessed, self.guessed,
                                                                                      self.last_guessed_type,
                                                                                      self.guessed_type) is True \
                        and self.guessed_type == self.correct_type:
                    self.in_correct = True

                if self.startOfChunk(self.last_correct, self.correct, self.last_correct_type,
                                     self.correct_type) is True:
                    self.found_correct[self.correct_type] += 1

                if self.startOfChunk(self.last_guessed, self.guessed, self.last_guessed_type,
                                     self.guessed_type) is True:
                    self.found_guessed[self.guessed_type] += 1

                if self.first_item != self.boundary:
                    if self.correct == self.guessed and self.guessed_type == self.correct_type:
                        self.correct_tags += 1
                    self.token_counter += 1

                self.last_guessed = self.guessed
                self.last_correct = self.correct
                self.last_guessed_type = self.guessed_type
                self.last_correct_type = self.correct_type

                if self.verbose > 1:
                    print("{0} {1} {2} {3} {4} {5} {6}".format(self.last_guessed, self.last_correct,
                                                               self.last_guessed_type, self.last_correct_type,
                                                               self.token_counter, len(self.found_correct.keys()),
                                                               len(self.found_guessed.keys())))

        if self.in_correct is True:
            self.correct_chunk[len(self.correct_chunk.keys())] = 0
            self.correct_chunk[self.last_correct_type] += 1

    def ComputeAccuracy(self):
        """Compute overall precision, recall and FB1 (default values are 0.0).
        Compute overall precision, recall and FB1 (default values are 0.0).
        Results:
            list: accuracy, precision, recall, FB1 float values.
        """
        if sum(self.found_guessed.values()) > 0:
            self.precision = 100 * sum(self.correct_chunk.values()) / float(sum(self.found_guessed.values()))
        if sum(self.found_correct.values()) > 0:
            self.recall = 100 * sum(self.correct_chunk.values()) / float(sum(self.found_correct.values()))
        if self.precision + self.recall > 0:
            self.FB1 = 2 * self.precision * self.recall / (self.precision + self.recall)

        overall = "processed {0} tokens with {1} phrases; found: {2} phrases; correct: {3}."
        overall = overall.format(self.token_counter, sum(self.found_correct.values()), sum(self.found_guessed.values()),
                                 sum(self.correct_chunk.values()))
        if self.verbose > 0:
            print(overall)

        self.accuracy = 100 * self.correct_tags / float(self.token_counter)
        if self.token_counter > 0 and self.verbose > 0:
            print("accuracy:  {0:0.2f}".format(self.accuracy))
            print("precision: {0:0.2f}".format(self.precision))
            print("recall:    {0:0.2f}".format(self.recall))
            print("FB1:       {0:0.2f}".format(self.FB1))

        return {"accuracy": self.accuracy, "precision": self.precision, "recall": self.recall, "FB1": self.FB1}

    def conlleval(self, predictions, groundtruth, words, infile):
        """Evaluate the results of one training iteration.

        Evaluate the results of one training iteration.  This now
        uses the native python port of the CoNLLeval perl script.
        It computes the accuracy, precision, recall and FB1 scores,
        and returns these as a dictionary.
        Args:
            predictions (list): Predictions from the network.
            groundtruth (list): Ground truth for evaluation.
            words (list): Corresponding words for de-referencing.
            infile:
        Returns:
            dict: Accuracy (accuracy), precisions (p), recall (r), and FB1 (f1) scores represented as floats.
            infile: The inputs written to file in the format understood by the conlleval.pl script and CoNLLeval python
                    port.
        """
        ofp = open(infile, "w")
        for sl, sp, sw in zip(groundtruth, predictions, words):
            ofp.write(u"BOS O O\n")
            for wl, wp, words in zip(sl, sp, sw):
                line = u"{0} {1} {2}\n".format(words, wl, wp)
                ofp.write(line)
            ofp.write(u"EOS O O\n\n")
        ofp.close()
        self.Evaluate(infile)
        return self.ComputeAccuracy()


if __name__ == "__main__":

    example = "{0} --infile".format(sys.argv[0])
    parser = argparse.ArgumentParser(description=example)
    parser.add_argument("--infile", "-i", help="Input CoNLLeval results file.", required=True)
    parser.add_argument("--raw", "-r", help="Accept raw result tags.", default=False, action="store_true")
    parser.add_argument("--delimiter", "-d", help="Token delimiter.", default=" ", type=str)
    parser.add_argument("--otag", "-ot", help="Alternative outside tag.", default="O", type=str)
    parser.add_argument("--boundary", "-b", help="Boundary tag.", default="-X-", type=str)
    parser.add_argument("--verbose", "-v", help="Verbose mode.", default=0, type=int)
    args = parser.parse_args()

    if args.verbose > 0:
        for key, val in args.__dict__.iteritems():
            print("{0}:  {1}".format(key, val))

    ce = CoNLLeval(verbose=args.verbose, raw=args.raw, delimiter=args.delimiter, otag=args.otag, boundary=args.boundary)
    ce.Evaluate(args.infile)
    results = ce.ComputeAccuracy()

    print()
    json.dumps(results, indent=4)
