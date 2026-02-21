from bson import ObjectId


def serialize_mongodb_doc(doc):
    if isinstance(doc, ObjectId):
        return str(doc)
    if not isinstance(doc, dict):
        return doc
    return {k: serialize_mongodb_doc(v) for k, v in doc.items()}
