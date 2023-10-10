import re
from pyspark.sql.types import StringType


def remove_scss_blocks(text):
    pattern_inline_style = r"style=\{\{[^}]+\}\}"
    pattern_class_style = r"""className=['"][^'"]+['"]"""
    pattern_styled_comp_0 = r"styled\.\w+`([\s\S]*?)`;"
    pattern_styled_comp_1 = r"styled\(\"?[a-zA-Z\.]*\"?\)`[\s\S]*?`;"
    text = re.sub(pattern_styled_comp_0, "", text)
    text = re.sub(pattern_styled_comp_1, "", text)
    text = re.sub(pattern_inline_style, "", text)
    text = re.sub(pattern_class_style, "", text)
    return text


def remove_jsx_stopwords(text):
    stopwords = [
        "accountid",
        "await",
        "block",
        "break",
        "branch",
        "button",
        "disabled",
        "classname",
        "color",
        "white",
        "position",
        "absolute",
        "left",
        "bottom",
        "background",
        "unset",
        "padding",
        "case",
        "catch",
        "class",
        "console",
        "const",
        "context",
        "continue",
        "debugger",
        "default",
        "delete",
        "div",
        "draft",
        "do",
        "else",
        "entries",
        "enum",
        "error",
        "export",
        "extends",
        "false",
        "finally",
        "flex",
        "for",
        "function",
        "gap",
        "height",
        "href",
        "http",
        "https",
        "icon",
        "if",
        "implements",
        "import",
        "in",
        "index",
        "init",
        "instanceof",
        "interface",
        "input",
        "item",
        "json",
        "img",
        "let",
        "log",
        "map",
        "new",
        "null",
        "obj",
        "object",
        "onclick",
        "package",
        "path",
        "private",
        "props",
        "protected",
        "public",
        "push",
        "return",
        "row",
        "section",
        "src",
        "start",
        "state",
        "stringify",
        "super",
        "svg",
        "switch",
        "tab",
        "this",
        "throw",
        "try",
        "true",
        "type",
        "typeof",
        "update",
        "undefined",
        "user",
        "url",
        "var",
        "var",
        "void",
        "while",
        "width",
        "with",
        "www",
        "xmlns",
        "yield",
    ]

    for word in stopwords:
        text = re.sub(r"\b" + re.escape(word) + r"\b", "", text)

    return text

# COMMAND ----------

import json
import re
from pyspark.sql import functions as F
from pyspark.sql.types import StringType


@F.udf(returnType=StringType())
def normalize_source_code_udf(doc):
    doc = re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        " ",
        doc,
    )
    doc = (
        doc.replace("\\n", " ").replace("\\t", " ").replace("\\r", " ").lower()
    )  # slightly different to normal string escaping as implemented in RecommenderNLPOperation utils...
    doc = re.sub(r"[^a-zA-Z0-9\s]", " ", doc, re.I | re.A)
    doc = re.sub(r"\s+", " ", doc)
    doc = doc.strip()
    return doc

# COMMAND ----------

import yake
from pyspark.sql.types import ArrayType, StringType

def extract_keywords(text, top_n=100):
    kw_extractor = yake.KeywordExtractor()
    keywords = kw_extractor.extract_keywords(text)
    return [kw[0] for kw in keywords[:top_n]]

# COMMAND ----------

from pyspark.sql.functions import udf

remove_styleblocks_udf = udf(remove_scss_blocks, StringType())
remove_stopwords_source_code_udf = udf(remove_jsx_stopwords, StringType())
extract_keywords_udf = udf(extract_keywords, ArrayType(StringType()))
