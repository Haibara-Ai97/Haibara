from flask import Flask, request, jsonify
from preprocess import crop, perspective_shift, hsv, remap, gray_Image, blur_image, resize, mean
from identify import bluemask, findcontour, countcenters, canny, houghlines, filterlines, DexiNed_edge
from count import countdistance, count_levelout, count_levelin


app = Flask(__name__)


@app.route('/count/countlevelin', methods=['GET', 'POST'])
def CountLevelin():

    if not request.is_json:
        return jsonify({"code": 400, "message": "Request data is not JSON"}), 400

    data = request.json

    if 'image' not in data or 'fixedParam' not in data or 'dynamicParam' not in data:
        return jsonify({"code": 401,
                        "message": "Missing 'image' or 'fixedParam' or 'dynamicParam' in JSON data"}), 401

    image = data['image']
    fixedParam = data['fixedParam']
    dynamicParam = data['dynamicParam']

    if 'level' not in fixedParam or 'map' not in fixedParam:
        return jsonify({"code": 402,
                        "message": "Missing parameters in 'fixedParam'"}), 402

    level = fixedParam['level']
    map = fixedParam['map']

    if 'original' not in dynamicParam or 'lines' not in dynamicParam:
        return jsonify({"code":402,
                        "message": "Missing 'original-image' or 'lines' in 'dynamicParam"})

    lines = dynamicParam['lines']
    original = dynamicParam['original']

    level_image, level = count_levelin(lines, original, level, map)

    return jsonify({
        "image": level_image,
        "dynamicParam": {
            "code": 200,
            "message": "Data received and processed",
            "finalResult": level,
        }
    }), 200


@app.route('/count/countlevelout', methods=['GET', 'POST'])
def CountLevelout():

    if not request.is_json:
        return jsonify({"code": 400, "message": "Request data is not JSON"}), 400

    data = request.json

    if 'image' not in data or 'fixedParam' not in data or 'dynamicParam' not in data:
        return jsonify({"code": 401,
                        "message": "Missing 'image' or 'fixedParam' or 'dynamicParam' in JSON data"}), 401

    image = data['image']
    fixedParam = data['fixedParam']
    dynamicParam = data['dynamicParam']

    if 'level' not in fixedParam or 'map' not in fixedParam:
        return jsonify({"code": 402,
                        "message": "Missing parameters in 'fixedParam'"}), 402

    level = fixedParam['level']
    map = fixedParam['map']

    if 'original' not in dynamicParam or 'lines' not in dynamicParam:
        return jsonify({"code":402,
                        "message": "Missing 'original-image' or 'lines' in 'dynamicParam"})

    lines = dynamicParam['lines']
    original = dynamicParam['original']

    level_image, level = count_levelout(lines, original, level, map)

    return jsonify({
        "image": level_image,
        "dynamicParam": {
            "code": 200,
            "message": "Data received and processed",
            "finalResult": {
                "figure": level,
                "unit": 'm',
                "description": "坝顶水尺水位",
            }
        }
    }), 200


@app.route('/identify/filterlines', methods=['GET', 'POST'])
def FilterLines():

    if not request.is_json:
        return jsonify({"code": 400, "message": "Request data is not JSON"}), 400

    data = request.json

    if 'image' not in data or 'fixedParam' not in data or 'dynamicParam' not in data:
        return jsonify({"code": 401,
                        "message": "Missing 'image' or 'fixedParam' or 'dynamicParam' in JSON data"}), 401

    image = data['image']
    fixedParam = data['fixedParam']
    dynamicParam = data['dynamicParam']

    if 'angledown' not in fixedParam or 'angleup' not in fixedParam or 'threshold' not in fixedParam:
        return jsonify({"code": 402,
                        "message": "Missing parameters in 'fixedParam'"}), 402

    angledown = fixedParam['angledown']
    angleup = fixedParam['angleup']
    threshold = fixedParam['threshold']

    if 'original' not in dynamicParam or 'lines' not in dynamicParam:
        return jsonify({"code":402,
                        "message": "Missing 'original-image' or 'lines' in 'dynamicParam"})

    original = dynamicParam['original']
    lines = dynamicParam['lines']

    filtered_image, filter_lines = filterlines(lines, original, angledown, angleup, threshold)

    return jsonify({
        "image": filtered_image,
        "dynamicParam": {
            "code": 200,
            "message": "Data received and processed",
            "lines": filter_lines,
            "original": original
        }
    }), 200


@app.route('/identify/houghlines', methods=['GET', 'POST'])
def Houghlines():

    if not request.is_json:
        return jsonify({"code": 400, "message": "Request data is not JSON"}), 400

    data = request.json

    if 'image' not in data or 'fixedParam' not in data or 'dynamicParam' not in data:
        return jsonify({"code": 401,
                        "message": "Missing 'image' or 'fixedParam' or 'dynamicParam' in JSON data"}), 401

    image = data['image']
    fixedParam = data['fixedParam']
    dynamicParam = data['dynamicParam']

    if 'hougha' not in fixedParam or 'houghb' not in fixedParam or 'houghc' not in fixedParam:
        return jsonify({"code": 402,
                        "message": "Missing 'hougha' or 'houghb' or 'houghc' in 'fixedParam'"}), 402

    hougha = fixedParam['hougha']
    houghb = fixedParam['houghb']
    houghc = fixedParam['houghc']

    if 'original' not in dynamicParam:
        return jsonify({"code":402,
                        "message": "Missing 'original-image' in 'dynamicParam"})

    original = dynamicParam['original']

    hough_image, lines = houghlines(image, original, hougha, houghb, houghc)

    return jsonify({
        "image": hough_image,
        "dynamicParam": {
            "code": 200,
            "message": "Data received and processed",
            "lines": lines,
            "original": original
        }
    }), 200


@app.route('/identify/dexiNed', methods=['GET', 'POST'])
def DexiNed():
    if not request.is_json:
        return jsonify({"code": 400, "message": "Request data is not JSON"}), 400

    data = request.json

    if 'image' not in data or 'fixedParam' not in data or 'dynamicParam' not in data:
        return jsonify({"code": 401,
                        "message": "Missing 'image' or 'fixedParam' or 'dynamicParam' in JSON data"}), 401

    image = data['image']
    fixedParam = data['fixedParam']
    dynamicParam = data['dynamicParam']

    if 'threshold' not in fixedParam:
        return jsonify({"code": 402,
                        "message": "Missing 'threshold' in 'fixedParam'"}), 402
    threshold = fixedParam['threshold']

    if 'original_size' not in dynamicParam or 'original' not in dynamicParam:
        return jsonify({"code": 402,
                        "message": "Missing 'original_size' or 'original' in 'dynamicParam'"}), 402

    original_size = dynamicParam['original_size']
    original = dynamicParam['original']

    dexined_image = DexiNed_edge(image, original_size, threshold)

    return jsonify({
        "image": dexined_image,
        "dynamicParam": {
            "code": 200,
            "message": "Data received and processed",
            "original":original
        }
    }), 200


@app.route('/preprocess/mean', methods=['GET', 'POST'])
def Mean():

    if not request.is_json:
        return jsonify({"code": 400, "message": "Request data is not JSON"}), 400

    data = request.json

    if 'image' not in data or 'fixedParam' not in data or 'dynamicParam' not in data:
        return jsonify({"code": 401,
                        "message": "Missing 'image' or 'fixedParam' or 'dynamicParam' in JSON data"}), 401

    image = data['image']
    fixedParam = data['fixedParam']
    dynamicParam = data['dynamicParam']

    if 'mean_bgr' not in fixedParam:
        return jsonify({"code": 402,
                        "message": "Missing 'mean_bgr' in 'fixedParam'"}), 402

    mean_bgr = fixedParam['mean_bgr']

    if 'original_size' not in dynamicParam or 'original' not in dynamicParam:
        return jsonify({"code": 402,
                        "message": "Missing 'original_size' or 'original' in 'dynamicParam'"}), 402

    original_size = dynamicParam['original_size']
    original = dynamicParam['original']

    mean_image = mean(image, mean_bgr)

    return jsonify({
        "image": mean_image,
        "dynamicParam": {
            "code": 200,
            "message": "Data received and processed",
            "original_size": original_size,
            "original": original
        }
    }), 200


@app.route('/preprocess/resize', methods=['GET', 'POST'])
def Resize():

    if not request.is_json:
        return jsonify({"code": 400, "message": "Request data is not JSON"}), 400

    data = request.json

    if 'image' not in data or 'fixedParam' not in data or 'dynamicParam' not in data:
        return jsonify({"code": 401,
                        "message": "Missing 'image' or 'fixedParam' or 'dynamicParam' in JSON data"}), 401

    image = data['image']
    fixedParam = data['fixedParam']
    dynamicParam = data['dynamicParam']

    if 'image_size' not in fixedParam:
        return jsonify({"code": 402,
                        "message": "Missing 'image_size' in 'fixedParam'"}), 402

    image_size = fixedParam['image_size']

    resize_image, original_size = resize(image, image_size)

    return jsonify({
        "image": resize_image,
        "dynamicParam": {
            "code": 200,
            "message": "Data received and processed",
            "original_size": original_size,
            "original": image
        }
    }), 200


@app.route('/identify/canny', methods=['GET', 'POST'])
def Canny():

    if not request.is_json:
        return jsonify({"code": 400, "message": "Request data is not JSON"}), 400

    data = request.json

    if 'image' not in data or 'fixedParam' not in data or 'dynamicParam' not in data:
        return jsonify({"code": 401,
                        "message": "Missing 'image' or 'fixedParam' or 'dynamicParam' in JSON data"}), 401

    image = data['image']
    fixedParam = data['fixedParam']
    dynamicParam = data['dynamicParam']

    if 'cannya' not in fixedParam or 'cannyb' not in fixedParam:
        return jsonify({"code": 402,
                        "message": "Missing 'cannya' or 'cannyb' in 'fixedParam'"}), 402

    cannya = fixedParam['cannya']
    cannyb = fixedParam['cannyb']

    if 'original' not in dynamicParam:
        return jsonify({"code":402,
                        "message": "Missing 'original-image' in 'dynamicParam"}), 402

    original = dynamicParam['original']

    canny_image = canny(image, cannya, cannyb)

    return jsonify({
        "image": canny_image,
        "dynamicParam": {
            "code": 200,
            "message": "Data received and processed",
            "original": original
        }
    }), 200


@app.route('/preprocess/blur', methods=['GET', 'POST'])
def Blur():

    if not request.is_json:
        return jsonify({"code": 400, "message": "Request data is not JSON"}), 400

    data = request.json

    if 'image' not in data or 'fixedParam' not in data or 'dynamicParam' not in data:
        return jsonify({"code": 401,
                        "message": "Missing 'image' or 'fixedParam' or 'dynamicParam' in JSON data"}), 401

    image = data['image']
    fixedParam = data['fixedParam']
    dynamicParam = data['dynamicParam']

    if 'original' not in dynamicParam:
        return jsonify({"code":402,
                        "message": "Missing 'original-image' in 'dynamicParam"}), 402

    original = dynamicParam['original']

    blurred_image = blur_image(original)

    return jsonify({
        "image": blurred_image,
        "dynamicParam": {
            "code": 200,
            "message": "Data received and processed",
            "original": original
        }
    }), 200


@app.route('/preprocess/gray', methods=['GET', 'POST'])
def Gray():

    if not request.is_json:
        return jsonify({"code": 400, "message": "Request data is not JSON"}), 400

    data = request.json

    if 'image' not in data or 'fixedParam' not in data or 'dynamicParam' not in data:
        return jsonify({"code": 401,
                        "message": "Missing 'image' or 'fixedParam' or 'dynamicParam' in JSON data"}), 401

    image = data['image']
    fixedParam = data['fixedParam']
    dynamicParam = data['dynamicParam']

    gray_image = gray_Image(image)

    return jsonify({
        "image": gray_image,
        "dynamicParam": {
            "code": 200,
            "message": "Data received and processed",
            "original": image
        }
    }), 200


@app.route('/preprocess/remap', methods=['GET', 'POST'])
def RemapImage():

    if not request.is_json:
        return jsonify({"code": 400, "message": "Request data is not JSON"}), 400

    data = request.json

    if 'image' not in data or 'fixedParam' not in data or 'dynamicParam' not in data:
        return jsonify({"code": 401,
                        "message": "Missing 'image' or 'fixedParam' or 'dynamicParam' in JSON data"}), 401

    image = data['image']
    fixedParam = data['fixedParam']
    dynamicParam = data['dynamicParam']

    if 'mapx' not in fixedParam or 'mapy' not in fixedParam:
        return jsonify({"code": 402,
                        "message": "Missing 'mapx' or 'mapy' in 'fixedParam'"}), 402

    mapx = fixedParam['mapx']
    mapy = fixedParam['mapy']

    remaped_image = remap(image, mapx, mapy)

    return jsonify({
        "image": remaped_image,
        "dynamicParam": {
            "code": 200,
            "message": "Data received and processed",
        }
    }), 200


@app.route('/identify/countdistance', methods=['GET', 'POST'])
def CountDistance():

    if not request.is_json:
        return jsonify({"code": 400, "message": "Request data is not JSON"}), 400

    data = request.json

    if 'image' not in data or 'fixedParam' not in data or 'dynamicParam' not in data:
        return jsonify({"code": 401,
                        "message": "Missing 'image' or 'fixedParam' or 'dynamicParam' in JSON data"}), 401

    image = data['image']
    fixedParam = data['fixedParam']
    dynamicParam = data['dynamicParam']

    if 'brick' not in fixedParam or 'weight' not in fixedParam:
        return jsonify({"code": 402,
                        "message": "Missing 'brick' or 'weight' in JSON data"}), 402

    brick = fixedParam['brick']
    weight = fixedParam['weight']

    if 'centers' not in dynamicParam:
        return jsonify({"code": 402,
                        "message": "Missing 'centers' in 'fixedParam'"}), 402

    centers = dynamicParam['centers']

    center_image, distance = countdistance(image, centers, brick, weight)

    return jsonify({
        "image": center_image,
        "dynamicParam": {
            "code": 200,
            "message": "Data received and processed",
            "finalReasult": {
                "figure": distance,
                "unit": "cm",
                "description": "Distance between two bricks"
            }
        }
    }), 200


@app.route('/identify/countcenters', methods=['GET', 'POST'])
def CountCenters():

    if not request.is_json:
        return jsonify({"code": 400, "message": "Request data is not JSON"}), 400

    data = request.json

    if 'image' not in data or 'fixedParam' not in data or 'dynamicParam' not in data:
        return jsonify({"code": 401,
                        "message": "Missing 'image' or 'fixedParam' or 'dynamicParam' in JSON data"}), 401

    image = data['image']
    fixedParam = data['fixedParam']
    dynamicParam = data['dynamicParam']

    if 'contours' not in dynamicParam:
        return jsonify({"code": 402,
                        "message": "Missing 'contours' in 'fixedParam'"}), 402

    contours = dynamicParam['contours']

    center_image, centers = countcenters(image, contours)

    return jsonify({
        "image": center_image,
        "dynamicParam": {
            "code": 200,
            "message": "Data received and processed",
            "centers": centers
        }
    }), 200


@app.route('/identify/findcontour', methods=['GET', 'POST'])
def FindContour():

    if not request.is_json:
        return jsonify({"code": 400, "message": "Request data is not JSON"}), 400

    data = request.json

    if 'image' not in data or 'fixedParam' not in data or 'dynamicParam' not in data:
        return jsonify({"code": 401,
                        "message": "Missing 'image' or 'fixedParam' or 'dynamicParam' in JSON data"}), 401

    image = data['image']

    fixedParam = data['fixedParam']
    dynamicParam = data['dynamicParam']

    if 'original' not in dynamicParam:
        return jsonify({"code": 402,
                        "message": "Missing 'mask' or 'original' in 'fixedParam'"}), 402

    original_image = dynamicParam['original']

    contours_image, contours = findcontour(original_image, image)

    return jsonify({
        "image": contours_image,
        "dynamicParam": {
            "code": 200,
            "message": "Data received and processed",
            "contours": contours
        }
    }), 200


@app.route('/identify/bluemask', methods=['GET', 'POST'])
def BlueMask():

    if not request.is_json:
        return jsonify({"code": 400, "message": "Request data is not JSON"}), 400

    data = request.json

    if 'image' not in data or 'fixedParam' not in data or 'dynamicParam' not in data:
        return jsonify({"code": 401,
                        "message": "Missing 'image' or 'fixedParam' or 'dynamicParam' in JSON data"}), 401

    image = data['image']
    fixedParam = data['fixedParam']
    dynamicParam = data['dynamicParam']

    if 'low' not in fixedParam or 'up' not in fixedParam:
        return jsonify({"code": 402,
                        "message": "Missing 'low' or 'up' in 'fixedParam'"}), 402

    low = fixedParam['low']
    up = fixedParam['up']

    if 'original' not in dynamicParam:
        return jsonify({"code": 402,
                        "message": "Missing 'original' in 'dynamicParam'"}), 402

    original_image = dynamicParam['original']

    masked_image = bluemask(original_image, low, up)

    return jsonify({
        "image": masked_image,
        "dynamicParam": {
            "code": 200,
            "message": "Data received and processed",
            "original": original_image
        }
    }), 200


@app.route('/preprocess/HSV_Image',methods=['GET', 'POST'])
def HSV_Image():

    if not request.is_json:
        return jsonify({"code": 400, "message": "Request data is not JSON"}), 400

    data = request.json

    if 'image' not in data or 'fixedParam' not in data or 'dynamicParam' not in data:
        return jsonify({"code": 401,
                        "message": "Missing 'image' or 'fixedParam' or 'dynamicParam' in JSON data"}), 401

    image = data['image']
    fixedParam = data['fixedParam']
    dynamicParam = data['dynamicParam']

    hsv_image = hsv(image)

    return jsonify({
        "image": hsv_image,
        "dynamicParam":  {
            "code": 200,
            "message": "Data received and processed",
            "original": image
        }
    }),200


@app.route('/preprocess/perspective_shift',methods=['GET', 'POST'])
def Perspective_shift():

    if not request.is_json:
        return jsonify({"code": 400, "message": "Request data is not JSON"}), 400

    data = request.json

    if 'image' not in data or 'fixedParam' not in data or 'dynamicParam' not in data:
        return jsonify({"code": 401,
                        "message": "Missing 'image' or 'fixedParam' or 'dynamicParam' in JSON data"}), 401

    image = data['image']
    fixedParam = data['fixedParam']
    dynamicParam = data['dynamicParam']

    if 'position' not in fixedParam:
        return jsonify({"code": 402,
                        "message": "Missing 'position in 'fixedParam'"}), 402

    position = fixedParam['position']

    perspective_image = perspective_shift(image, position)

    return jsonify({
        "image": perspective_image,
        "dynamicParam":  {
            "code": 200,
            "message": "Data received and processed",
        }
    }),200


@app.route('/preprocess/crop',methods=['GET', 'POST'])
def Crop():

    if not request.is_json:
        return jsonify({"code": 400, "message": "Request data is not JSON"}), 400

    data = request.json

    if 'image' not in data or 'fixedParam' not in data or 'dynamicParam' not in data:
        return jsonify({"code": 401,
                        "message": "Missing 'image' or 'fixedParam' or 'dynamicParam' in JSON data"}), 401

    image = data['image']
    fixedParam = data['fixedParam']
    dynamicParam = data['dynamicParam']

    if 'position' not in fixedParam:
        return jsonify({"code": 402,
                        "message": "Missing 'position in 'fixedParam'"}), 402

    position = fixedParam['position']

    cropped_image = crop(image, position)

    return jsonify({
        "image": cropped_image,
        "dynamicParam":  {
            "code": 200,
            "message": "Data received and processed",
        }
    }),200


@app.route('/test')
def home():
    return "Hello"


if __name__ == '__main__':
    app.run(debug=True)