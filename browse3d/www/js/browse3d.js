// based on https://threejsfundamentals.org/threejs/lessons/threejs-multiple-scenes.html

import * as THREE from 'https://unpkg.com/three@0.118.3/build/three.module.js';

import { GUI } from 'https://unpkg.com/three@0.118.3/examples/jsm/libs/dat.gui.module.js';

import { OrbitControls } from 'https://unpkg.com/three@0.118.3/examples/jsm/controls/OrbitControls.js';
import { OBJLoader } from 'https://unpkg.com/three@0.118.3/examples/jsm/loaders/OBJLoader.js';
import { MTLLoader } from 'https://unpkg.com/three@0.118.3/examples/jsm/loaders/MTLLoader.js';
import { GLTFLoader } from 'https://unpkg.com/three@0.118.3/examples/jsm/loaders/GLTFLoader.js';
import { PLYLoader } from 'https://unpkg.com/three@0.118.3/examples/jsm/loaders/PLYLoader.js';


function main() {
    // config
    const config = {
        animate: false,
        cycleKeypoints: false,
        
        // airplane
        cameraPosition: [0.5, 2.5, 1.25],
        cameraPosition: [1.7, 3, 2.25],
        scale: 0.9,
        pointSize: 0.09,
    }

    const canvas = document.querySelector('#canvas');
    const renderer = new THREE.WebGLRenderer({canvas, alpha: true, antialias: true, preserveDrawingBuffer: true});

    const sceneElements = [];

    const OBJColor = 0x2e81d4;
    const OBJOpacity = config.OBJOpacity;

    var pointsCounter = 0;

    function makeScene(elem) {


        const scene = new THREE.Scene();
        const fov = 45;
        const aspect = 2;  // the canvas default
        const near = 0.01;
        const far = 100;
        const camera = new THREE.PerspectiveCamera(fov, aspect, near, far);
        
        let cameraLookAt = [0, 0, 0];
        
        let cameraPosition = config.cameraPosition;
        let scale = config.scale;

        cameraPosition = cameraPosition.map(x => x * scale);
        camera.position.set(... cameraPosition);
        camera.lookAt(... cameraLookAt);
        scene.add(camera);

        scene.add(camera);

        const controls = new OrbitControls(camera, elem);
        controls.minDistance = near;
        controls.maxDistance = far;
        controls.enableDamping = false;
        controls.autoRotate = false;
        controls.addEventListener('end', () => {
            let vector = new THREE.Vector3( 0, 0, - 1 );
            vector.applyQuaternion( camera.quaternion );
            console.log(
                'postion: ' + camera.position['x'] + ', ' + camera.position['y'] + ', ' + camera.position['z'] + 
                ' rotation: ' + camera.rotation['x'] + ', ' + camera.rotation['y'] + ', ' + camera.rotation['z'] + 
                ' cameraLookAt: ' + vector.toArray());
        });

        let ambientLight = new THREE.AmbientLight( 0xffffff, 0.2 );
        scene.add( ambientLight );

        let hemisphereLight = new THREE.HemisphereLight( 0xffffff, 0x646464, 0.55 );
        hemisphereLight.position.set( 0, 20, 0 );
        scene.add( hemisphereLight );

        let hemisphereLight2 = new THREE.HemisphereLight( 0xffffff, 0x646464, 0.2 );
        hemisphereLight2.position.set( 0, 0, -10 );
        scene.add( hemisphereLight2 );

        let spotLight = new THREE.SpotLight( 0xffffff, 1 );
        spotLight.position.set( -5, 5, 1 );
        spotLight.angle = 0.15;
        spotLight.penumbra = 1;
        spotLight.decay = 2;
        spotLight.distance = 200;
        spotLight.intensity = 0.2;

        spotLight.castShadow = true;
        spotLight.shadow.mapSize.width = 512;
        spotLight.shadow.mapSize.height = 512;
        spotLight.shadow.camera.near = 10;
        spotLight.shadow.camera.far = 200;
        spotLight.shadow.focus = 1;
        scene.add( spotLight );

        let spotLight2 = new THREE.SpotLight( 0xffffff, 1 );
        spotLight2.position.set( -1, 5, 5 );
        spotLight2.angle = 0.15;
        spotLight2.penumbra = 1;
        spotLight2.decay = 2;
        spotLight2.distance = 200;
        spotLight2.intensity = 0.26;

        spotLight2.castShadow = true;
        spotLight2.shadow.mapSize.width = 512;
        spotLight2.shadow.mapSize.height = 512;
        spotLight2.shadow.camera.near = 10;
        spotLight2.shadow.camera.far = 200;
        spotLight2.shadow.focus = 1;
        scene.add( spotLight2 );

        return {scene, camera, controls};
    }

    function initScene(elem) {
        const {scene, camera, controls} = makeScene(elem);
        var group = new THREE.Object3D();
        scene.add(group);
        var fn = {
            customFn: () => {},
            animate(time, rect) {
                this.customFn(time, group);
                camera.aspect = rect.width / rect.height;
                camera.updateProjectionMatrix();
                controls.update();
                renderer.render(scene, camera)
            }
        };
        sceneElements.push({elem, fn});
        return [scene, group, fn];
    }
    
    function addPoint(scene, position, name=null) {
        if (name == null) {
            name = pointsCounter.toString().padStart(3, '0');
        }

        var color = hashStringToColor(name);

        var geometry = new THREE.SphereGeometry(config.pointSize, 32, 32 );
        var material = new THREE.MeshPhongMaterial({
            color: color, flatShading: false, emissiveIntensity: 0.5});
        var m = new THREE.Mesh( geometry, material );
        m.position.copy( position );

        var point = {
            name: name,
            color: color,
            m: m
        }
        scene.add( m );

        pointsCounter++ ;
        
        return m;
    }
    
    function fetchKeypoints(url, scene, fn) {
        fetch(url).then(function(response) {
            return response.text()
        }).then(function(text) {
            var points = loadKeypoints(text, scene);
            if (config.rotate) {
                fn.customFn = (time, group) => {
                    // cycle thourgh points
                    if (config.cycleKeypoints) {
                        const interval = 2;
                        const period = points.length * interval;
                        const realativeTime = time % period;
                        for (const point of points) {
                            point.material.emissive.set(0x000000);
                        }
                        const index = Math.floor(realativeTime / interval);
                        points[index].material.emissive.set(0xaaaaaa);
                    }

                    group.rotation.y = time * .2 - 2;
                }
            }
        });
    }


    function loadKeypoints(string, scene) {
        var lines = string.split('\n');
        var points = [];
        for (const line of lines) {
            var split = line.split(' ');
            if (split.length != 4) 
                continue;
            var name = split[0];
            var position = new THREE.Vector3(split[1], split[2], split[3]);
            var point = addPoint(scene, position, name=name);
            points.push(point);
        }
        return points;
    }


    function fetchPointcloud(url, format, scene) {
        fetch(url).then(function(response) {
            return response.text()
        }).then(function(text) {
            scene.add(parsePointcloud(text, format));
        });
    }

    
    function createBasicPointsObject(position, color, alpha, circle=false) {
        var geometry = new THREE.BufferGeometry();

        geometry.setAttribute( 'position', new THREE.Float32BufferAttribute( position, 3 ) );
        if ( color.length > 0 ) geometry.setAttribute( 'color', new THREE.Float32BufferAttribute( color, 3 ) );
        if ( alpha.length > 0 ) geometry.setAttribute( 'alpha', new THREE.Float32BufferAttribute( alpha, 1 ) );

        geometry.computeBoundingSphere();
        
        if (circle) {
            const sprite = new THREE.TextureLoader().load( 'https://raw.githubusercontent.com/mrdoob/three.js/master/examples/textures/sprites/disc.png' );
            var material = new THREE.PointsMaterial( { size: 0.05, sizeAttenuation: true, map: sprite, alphaTest: 0.5, transparent: true } );
        } else {
            var material = new THREE.PointsMaterial( { size: 0.002, alphaTest: 0.5, transparent: true} );
        }

        if ( color.length > 0 ) {
			material.vertexColors = true;
		} else {
			material.color.setHex( 0xff0000 );
		}

        var mesh = new THREE.Points( geometry, material );
        
        return mesh
    }


    function createPoint(position, color, alpha, size=0.05) {
        var color = new THREE.Color(1, 0, 0);
        var alpha = (alpha + 0.2) / 1.2;

        var geometry = new THREE.SphereGeometry(size, 16, 16 );
        var material = new THREE.MeshStandardMaterial({
            color: color, opacity: alpha, transparent: true} );
        var m = new THREE.Mesh( geometry, material );
        m.position.copy(new THREE.Vector3(position[0], position[1], position[2]));
        return m;
    }


    function createPointsObject(position, color, alpha) {
        var object_mesh = new THREE.Object3D;
        var position = listToMatrix(position, 3)
        var color = listToMatrix(color, 3)
        for (let i = 0; i < position.length; i++) {
            if (alpha[i] > 0.05) {
                object_mesh.add(createPoint(position[i], color[i], alpha[i]));
            }
        }
        return object_mesh
    }


    function parsePointcloud(string, format, circle=false, mild_colors=false, basic=true) {
        // based on https://github.com/mrdoob/three.js/blob/master/examples/jsm/loaders/PCDLoader.js
        var lines = string.split('\n');
        var position = [];
        var color = [];
        var alpha = [];
        for (const line of lines) {
            var split = line.split(' ');
            position.push(parseFloat(split[0]));
            position.push(parseFloat(split[1]));
            position.push(parseFloat(split[2]));
            if (format == 'xyzrgb' || format == 'xyzrgba') {
                var [r, g, b] = split.slice(3, 6).map(x => parseFloat(x));
                if (mild_colors) {
                    [r, g, b] = [r, g, b].map(x => x * 255);
                    var [h, s, l] = RGBToHSL(r, g, b);
                    s = 50;
                    l = 60;
                    [r, g, b] = HSLToRGB(h, s, l).map(x => x / 255);
                }
                color.push(r, g, b);
            } else {
                color.push(1, 0, 0);
            }
            if (format == 'xyzrgba') {
                alpha.push(parseFloat(split[6]));
            } else {
                alpha.push(0.8);
            }            

        }
        
        if (basic) {
            return createBasicPointsObject(position, color, alpha);
        } else {
            return createPointsObject(position, color, alpha);
        }
    }


    function loadModel(url, scene, secondary=false, mtl=null) {
        var ext = url.split('.').pop();
        if (ext == 'obj') {
            if (mtl == null) {
                new OBJLoader().load(url, onLoad);
            } else {
                new MTLLoader()
                    .setMaterialOptions({side: THREE.DoubleSide})
                    .load(mtl, (materials) => {
                        materials.preload();
                        new OBJLoader()
                            .setMaterials(materials)
                            .load(url, (object) => {
                                object.traverse(function(child) {
                                    if (child instanceof THREE.Mesh) {
                                        child.castShadow = true;
                                        child.material.side = THREE.DoubleSide;
                                        child.material.flatShading = true;
                                    }
                                });
                                scene.add(object);
                            });
                    });
            }
        } else if (ext == 'glb') {
            GLTFLoader().load(url, onLoad);
        }
        
        function onLoad(object) {
            // for
            if (object.hasOwnProperty('scene')) {
                object = object.scene;
            }
            if (secondary) {
                material = new THREE.MeshStandardMaterial({
                    color: 0x0055ff, flatShading: true, opacity: 0.2, transparent: true});
            } else {
                var material = new THREE.MeshPhongMaterial({
                    color: OBJColor, flatShading: true, dithering: true, opacity: OBJOpacity, transparent: true});
            }

            object.traverse(function(child) {
                if (child instanceof THREE.Mesh) {
                    child.material = material;
                    child.castShadow = true;
                    child.material.side = THREE.DoubleSide;
                }
            });

            // merge
            var mesh = object.children[0] ;
                
            scene.add(mesh);
        }
    }

    function loadPLY(url, scene) {
        var loader = new PLYLoader();
        loader.load(url, onLoad);
        
        function onLoad(geometry) {
            
            // add wireframe
            var geo = new THREE.EdgesGeometry(geometry); // or WireframeGeometry
            var mat = new THREE.LineBasicMaterial({color: 0x000000, linewidth: 2});
            var wireframe = new THREE.LineSegments(geo, mat);

            scene.add(wireframe);
        }
    }

    document.querySelectorAll('[data-object]').forEach((elem) => {
        var [_, group, fn] = initScene(elem);
        if (elem.dataset.hasOwnProperty('mtl')) {
            loadModel(elem.dataset.obj, group, false, elem.dataset.mtl);
        } else if (elem.dataset.hasOwnProperty('obj')) {
            loadModel(elem.dataset.obj, group);
        }
        if (elem.dataset.hasOwnProperty('ply')) {
            loadPLY(elem.dataset.ply, group);
        }
        if (elem.dataset.hasOwnProperty('kpts')) {
            fetchKeypoints(elem.dataset.kpts, group, fn=fn);
        }
        if (elem.dataset.hasOwnProperty('pts')) {
            fetchPointcloud(elem.dataset.pts, 'xyz', group);
        }
        if (elem.dataset.hasOwnProperty('xyzrgb')) {
            fetchPointcloud(elem.dataset.xyzrgb, 'xyzrgb', group);
        }
        if (elem.dataset.hasOwnProperty('xyzrgba')) {
            fetchPointcloud(elem.dataset.xyzrgba, 'xyzrgba', group);
        }
        if (elem.dataset.hasOwnProperty('obj_second')) {
            loadOBJ(elem.dataset.obj_second, group, secondary=true);
        }
    });

    function resizeRendererToDisplaySize(renderer) {
        const canvas = renderer.domElement;
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        const needResize = canvas.width !== width || canvas.height !== height;
        if (needResize) {
        renderer.setSize(width, height, false);
        }
        return needResize;
    }

    const clearColor = new THREE.Color('#000');
    function render(time) {
        time *= 0.001;

        resizeRendererToDisplaySize(renderer);

        renderer.setScissorTest(false);
        renderer.setClearColor(clearColor, 0);
        renderer.clear(true, true);
        renderer.setScissorTest(true);

        const transform = `translateX(${window.scrollX}px)`;
        renderer.domElement.style.transform = transform;
        
        if (config.animate) {
            const interval = 1/30;
            const length = sceneElements.length;
            const period = length * interval;
            const realativeTime = time % period;
            const direction = Math.floor((time % (2 * period) / period));
            let index = Math.floor(realativeTime / interval);
            index = direction * index + (1 - direction) * (length - index - 1);
            console.log(direction + ' ' + index);
            var sceneElement = sceneElements[index];

            const {elem, fn} = sceneElement;
            // get the viewport relative position of this element
            const rect = sceneElements[0].elem.getBoundingClientRect();
            const {left, right, top, bottom, width, height} = rect;

            const isOffscreen =
                bottom < 0 ||
                top > renderer.domElement.clientHeight ||
                right < 0 ||
                left > renderer.domElement.clientWidth;

            if (!isOffscreen) {
                const positiveYUpBottom = renderer.domElement.clientHeight - bottom;
                renderer.setScissor(left, positiveYUpBottom, width, height);
                renderer.setViewport(left, positiveYUpBottom, width, height);

                fn.animate(time, rect);
            }
        } else {
            for (const {elem, fn} of sceneElements) {
                // get the viewport relative position of this element
                const rect = elem.getBoundingClientRect();
                const {left, right, top, bottom, width, height} = rect;
    
                const isOffscreen =
                    bottom < 0 ||
                    top > renderer.domElement.clientHeight ||
                    right < 0 ||
                    left > renderer.domElement.clientWidth;
    
                if (!isOffscreen) {
                    const positiveYUpBottom = renderer.domElement.clientHeight - bottom;
                    renderer.setScissor(left, positiveYUpBottom, width, height);
                    renderer.setViewport(left, positiveYUpBottom, width, height);
    
                    fn.animate(time, rect);
                }
            }
        }
        

        requestAnimationFrame(render);
    }

    var params = {
        screenshot: function () {saveAsImage(renderer);},
    };

    var gui = new GUI();
    gui.add( params, 'screenshot' );
    gui.open();

    requestAnimationFrame(render);
}



function hashStringToColor(string) {
    var hash = hashString(string);
    var color = hash % 0xffffff;
    return color;
}

function hashString(str) {
    // based on: https://github.com/darkskyapp/string-hash/blob/master/index.js
    // var hash = 0, i, chr;
    // for (i = 0; i < string.length; i++) {
    // 	chr   = string.charCodeAt(i);
    // 	hash  = ((hash << 5) - hash) + chr;
    // 	hash |= 0; // Convert to 32bit integer
    // }
    // return hash;
    var hash = 5381,
    i = str.length;

    while(i) {
        hash = (hash * 5381) ^ str.charCodeAt(--i);
    }

    /* JavaScript does bitwise operations (like XOR, above) on 32-bit signed
    * integers. Since we want the results to be always positive, convert the
    * signed int to an unsigned by doing an unsigned bitshift. */
    return hash >>> 0;
}

function sphericalToCart(r, azimuth, elevation) {
    // inputs in degrees
    var x = r * Math.sin(elevation * Math.PI / 180) * Math.sin(azimuth * Math.PI / 180);
    var y = r * Math.sin(elevation * Math.PI / 180) * Math.cos(azimuth * Math.PI / 180);
    var z = r * Math.cos(elevation * Math.PI / 180);
    return {x, y, z};
}


function RGBToHSL(r,g,b) {
    // From https://css-tricks.com/converting-color-spaces-in-javascript/

    // Make r, g, and b fractions of 1
    r /= 255;
    g /= 255;
    b /= 255;

    // Find greatest and smallest channel values
    let cmin = Math.min(r,g,b),
        cmax = Math.max(r,g,b),
        delta = cmax - cmin,
        h = 0,
        s = 0,
        l = 0;

    // Calculate hue
    // No difference
    if (delta == 0)
        h = 0;
    // Red is max
    else if (cmax == r)
        h = ((g - b) / delta) % 6;
    // Green is max
    else if (cmax == g)
        h = (b - r) / delta + 2;
    // Blue is max
    else
        h = (r - g) / delta + 4;

    h = Math.round(h * 60);
    
    // Make negative hues positive behind 360°
    if (h < 0)
        h += 360;
    
    // Calculate lightness
    l = (cmax + cmin) / 2;
  
    // Calculate saturation
    s = delta == 0 ? 0 : delta / (1 - Math.abs(2 * l - 1));
      
    // Multiply l and s by 100
    s = +(s * 100).toFixed(1);
    l = +(l * 100).toFixed(1);
  
    return [h, s, l];
}


function HSLToRGB(h,s,l) {
    // Must be fractions of 1
    s /= 100;
    l /= 100;
  
    let c = (1 - Math.abs(2 * l - 1)) * s,
        x = c * (1 - Math.abs((h / 60) % 2 - 1)),
        m = l - c/2,
        r = 0,
        g = 0,
        b = 0;

    if (0 <= h && h < 60) {
        r = c; g = x; b = 0;
        } else if (60 <= h && h < 120) {
        r = x; g = c; b = 0;
        } else if (120 <= h && h < 180) {
        r = 0; g = c; b = x;
        } else if (180 <= h && h < 240) {
        r = 0; g = x; b = c;
        } else if (240 <= h && h < 300) {
        r = x; g = 0; b = c;
        } else if (300 <= h && h < 360) {
        r = c; g = 0; b = x;
        }
        r = Math.round((r + m) * 255);
        g = Math.round((g + m) * 255);
        b = Math.round((b + m) * 255);
    
        return [r, g, b];
}


function saveAsImage(renderer) {
    var imgData;

    try {
        var strMime = "image/png";
        var strDownloadMime = "image/octet-stream";
        imgData = renderer.domElement.toDataURL(strMime);

        saveFile(imgData.replace(strMime, strDownloadMime), "screenshot.png");

    } catch (e) {
        console.log(e);
        return;
    }

}

var saveFile = function (strData, filename) {
    var link = document.createElement('a');
    if (typeof link.download === 'string') {
        document.body.appendChild(link); //Firefox requires the link to be in the body
        link.download = filename;
        link.href = strData;
        link.click();
        document.body.removeChild(link); //remove the link when done
    } else {
        location.replace(uri);
    }
}

function listToMatrix(list, elementsPerSubArray) {
    // https://stackoverflow.com/questions/4492385/how-to-convert-simple-array-into-two-dimensional-array-matrix-with-javascript
    var matrix = [], i, k;

    for (i = 0, k = -1; i < list.length; i++) {
        if (i % elementsPerSubArray === 0) {
            k++;
            matrix[k] = [];
        }

        matrix[k].push(list[i]);
    }

    return matrix;
}

function rgbToNumber(r, g, b) {
    return (r << 16) + (g << 8) + b;
}

main();
