import bs4
from bs4 import BeautifulSoup, NavigableString
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer


class FeatureExtraction():
    def __init__(self, html_document, keywords):
        self.html_document = html_document
        self.keywords = keywords

        soup = BeautifulSoup(html_document, "html.parser")
        self.soup_object = soup

        self.feature_vector = np.zeros(18)

    def __repr__(self):
        print(f"HTML document: {len(self.html_document) != 0}")
        print(f"Keywords: {self.keywords}")

    def extract_navigable_strings(self, tag):
        for child in tag:
            if isinstance(child, NavigableString):
                child.extract()
            elif child.name:
                self.extract_navigable_strings(child)

    def feature_extraction_meta_tag(self):
        """
        Meta tag features: 
        Application name, 
        Keywords, 
        Author, 
        Description, 
        Description tag length (<= 160) 
        Generator, 
        Viewport, 
        Date, 
        """

        # get features from meta tag
        for meta_tag in self.soup_object.find_all("meta"):
            attributes = meta_tag.attrs
            has_description = False
            if "name" in attributes and "content" in attributes:
                # check for null content
                if attributes["content"] == "":
                    continue

                # check for each description necessary for meta tag
                if attributes["name"] == "application-name":
                    self.feature_vector[0] += 1
                elif attributes["name"] == "keywords":
                    self.feature_vector[1] += 1
                elif attributes["name"] == "author":
                    self.feature_vector[2] += 1
                elif attributes["name"] == "description":
                    self.feature_vector[3] += 2
                    has_description = True
                    if len(attributes["content"]) <= 160:
                        self.feature_vector[4] += 1
                elif attributes["name"] == "generator":
                    self.feature_vector[5] += 1
                elif attributes["name"] == "viewport":
                    self.feature_vector[6] += 1
                elif attributes["name"] == "date":
                    self.feature_vector[7] += 1
                elif attributes["name"] == "lang":
                    self.feature_vector[8] -= 1

        if not has_description:
            self.feature_vector[3] -= 2
        return

    def feature_extraction_head_tag(self):
        """
        Head tag features: 
        Has Title tag
        Title tag must contain keywords 
        """
        self.feature_vector[9] = 1

        head_tag = self.soup_object.find("head")
        title_tag_content = ""
        found_title = False
        kw_count = 0

        i = 0
        for head_tag_children in head_tag.children:
            if head_tag_children == "title":
                self.feature_vector[9] -= i
                title_tag_content = head_tag_children.text
                found_title = True
            i += 1

        for kw in self.keywords:
            if kw in title_tag_content:
                if kw_count > 3:
                    self.feature_vector[10] -= 1
                else:
                    self.feature_vector[10] += 1

        if not found_title:
            self.feature_vector[9] -= 3

    def feature_extraction_html_tag(self):
        """
        HTML tag features: 
        lang
        head 
        body 
        footer
        """
        for i in range(11, 15):
            self.feature_vector[i] = 1

        if self.soup_object.find("html").attrs == None:
            self.feature_vector[11] -= 1

        if self.soup_object.find("head").attrs == None:
            self.feature_vector[12] -= 1

        if self.soup_object.find("body").attrs == None:
            self.feature_vector[13] -= 1

        if self.soup_object.find("footer").atrrs == None:
            self.feature_vector[14] -= 1

    def feature_extraction_misc(self):
        """
        Misc features: 
        Anchor text, 
        Heading tags,
        Alt text images 
        """

        # anchor tag
        self.feature_vector[15] = 1
        for anchor_tag in self.soup_object.find_all("a"):
            if anchor_tag.text == "":
                self.feature_vector[15] -= 1

        # chekc for title
        self.feature_vector[16] = 2
        for i in range(6, 0, 1):
            if self.soup_object.find(f"h{i}") != None:
                for i in range(i, 0, 1):
                    if self.soup_object.find(f"h{i}") == None:
                        self.feature_vector[16] -= 1
                return

        # check for alt text
        self.feature_vector[17] = 2
        for img_tag in self.soup_object.find_all("img"):
            if img_tag.attrs == None:
                self.feature_vector[17] -= 1
            elif "alt" not in img_tag.attrs:
                self.feature_vector[17] -= 1
            elif img_tag["alt"] == "":
                self.feature_vector[17] -= 1

    def start(self):
        self.feature_extraction_html_tag()
        self.feature_extraction_head_tag()
        self.feature_extraction_meta_tag()
        self.feature_extraction_misc()

    def get_feature_vector(self):
        norm = np.linalg.norm(self.feature_vector)
        return self.feature_vector / norm
