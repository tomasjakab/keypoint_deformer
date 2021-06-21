
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes">
    <title>{{title}}</title>
    <style>
    #canvas {
      position: absolute;
      left: 0;
      top: 0;
      % if not opt.vertical:
      width: {{opt.im_size[0] * len(samples) +20}}px;
      % else:
      width: {{opt.im_size[0] * len(visuals) +20}}px;
      % end
      % if not opt.vertical:
      height: {{opt.im_size[1] * 1.2 * len(visuals)}}px;
      % else:
      height: {{opt.im_size[1] * 1.2 * len(samples)}}px;
      % end
      display: block;
      z-index: -1;
    }
    *[data-object] {
      display: inline-block;
      width: {{opt.im_size[0]}}px;
      height: {{opt.im_size[1]}}px;
    }
    td {
      width: {{opt.im_size[0]}}px;
      word-break: break-word;
    }
    p {
      margin: 1em auto;
      max-width: 500px;
      font-size: xx-large;
    }
    </style>
  </head>
  <body>
    <canvas id="canvas"></canvas>
    {{title}}
    <table style="table-layout: fixed; border: 0;">
      % if not opt.vertical:
        % for field in visuals:
          <tr>
            <td>
              % if 'title' in field:
                    {{field['title']}}
              % end
            </td>
          </tr>

          <tr>
          % for sample in samples:
            <td>
              <span>{{sample}}</span>
            </td>
          % end
          </tr>

          <tr>
            % for sample in samples:
              <td>
                <span data-object\\
                %for type, file in field['visuals'].items():
                data-{{type}}="data/{{sample}}/{{file}}"\\
                %end
                ></span>
              </td>
            % end
          </tr>

        % end
      % else:
        % for sample in samples:
          <tr>
            <td>
              <span>{{sample}}</span>
            </td>
          </tr>

          <tr>
          % for field in visuals:
            <td>
              % if 'title' in field:
                    {{field['title']}}
              % end
            </td>
          % end
          </tr>
          
          <tr>
          % for field in visuals:
            <td>
              <span data-object\\
              %for type, file in field['visuals'].items():
              data-{{type}}="data/{{sample}}/{{file}}"\\
              %end
              ></span>
            </td>
          % end
          </tr>

        % end
      % end
    </table>
  </body>
  <script src="js/browse3d.js" type="module"></script>

  <script type="x-shader/x-vertex" id="vertexshader">

      attribute float alpha;

      varying float vAlpha;

      void main() {

          vAlpha = alpha;

          vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );

          if (alpha > 0.1) {
              gl_PointSize = 16.0;
          } else {
              gl_PointSize = 0.0;
          }

          gl_Position = projectionMatrix * mvPosition;

      }

  </script>

  <script type="x-shader/x-fragment" id="fragmentshader">

      uniform vec3 color;

      varying float vAlpha;

      void main() {

          gl_FragColor = vec4( color, vAlpha );

      }

  </script>


</html>

