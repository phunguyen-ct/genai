def beautify(answer):
    divider = "-" * 50
    references = ''
    if (len(answer.source_nodes) > 0):
        for node in answer.source_nodes:
            # Extract metadata from the node
            name = node.node.metadata.get('Name', 'Name not found')
            price = node.node.metadata.get('Price', 'Price not found')
            location = node.node.metadata.get(
                'Location', 'Location not found')
            url = node.node.metadata.get('URL', 'URL not found')
            description = node.node.metadata.get(
                'Description', 'Description not found')
            area = node.node.metadata.get('Area', 'Area not found')

            references = references + f"<li>{name}</li>"
            references = references + f"<li>Giá: {price}</li>"
            references = references + f"<li>Khu vực: {location}</li>"
            references = references + f"<li>URL: {url}</li>"
            references = references + f"<li>Mô tả: {description}</li>"
            references = references + f"<li>Diện tích: {area}</li>"
            references = references + divider + "<br>"

        response = f"<pre style='text-wrap: pretty;'><p>{
            answer}</p><br><b>Tham khảo thêm:</b><br><ul>{references}</ul></pre>"
    else:
        response = f"<pre style='text-wrap: pretty;'><p>{
            answer}</p></pre>"

    return response
