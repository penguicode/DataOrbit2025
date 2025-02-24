
const request = new XMLHttpRequest();
request.open("GET", "low_quality.geojson", false); // `false` makes the request synchronous
request.send(null);

var data = JSON.parse(request.response).features;

console.log(data[1])



let circle_center = [-121.961815002484, 39.8297044165437]
let circle_radius = 0.01;

console.log(get_nearby_fires(circle_center, circle_radius))

function get_nearby_fires(circle_center, circle_radius) {
   let collisions = []
   for (const features of data) {
      geometry = features.geometry
      //console.log(features.properties["FIRE_NAME"])
      //console.log(geometry.type)
      if (geometry.type == 'Polygon') {
         let polygon = geometry.coordinates[0]
         let polygon_half = Math.floor(polygon.length/2);
         //console.log(polygon_half)
         let center = [(polygon[0][0] + polygon[polygon_half][0])/2, (polygon[0][1]+polygon[polygon_half][1])/2]
         if (dist(center, circle_center) <= circle_center) break;
         for (const points of polygon) {
            if (dist(points, circle_center) <= circle_radius) {
               break;
            }
         }
         let point1 = polygon[0]
         if (lineCircle(polygon[0], polygon[polygon.length-1], circle_center, circle_radius)) {
            collisions.push(features)
         }else {
            for (let i = 1; i < polygon.length; i++) {
               let point2 = polygon[i]
               if (lineCircle(point1, point2, circle_center, circle_radius)) {
                  collisions.push(features)
                  break;
               }
               point1 = point2
            }
         }
      } else {
         //console.log(geometry.coordinates)
         for (const polygon of geometry.coordinates[0]) {
            let polygon_half = Math.floor(polygon.length/2);
            //console.log(polygon_half)
            let center = [(polygon[0][0] + polygon[polygon_half][0])/2, (polygon[0][1]+polygon[polygon_half][1])/2]
            if (dist(center, circle_center) <= circle_center) break;
            let hasCollide = false;
   
            for (const points of polygon) {
               if (dist(points, circle_center) <= circle_radius) {
                  collisions.push(features)
                  hasCollide = true;
                  break;
               }
            }
            if (hasCollide) break;
            let point1 = polygon[0]
            if (lineCircle(polygon[0], polygon[polygon.length-1], circle_center, circle_radius)) {
               collisions.push(features)
               hasCollide = true;
               break;
            } else {
               for (let i = 1; i < polygon.length; i++) {
                  let point2 = polygon[i]
                  if (lineCircle(point1, point2, circle_center, circle_radius)) {
                     collisions.push(features)
                     hasCollide = true;
                     break;
                  }
                  point1 = point2
               }
            }
            if (hasCollide) break;
         }
      }
   
   }
   
   // LINE/CIRCLE
   function lineCircle(point1, point2, circle_center, circle_radius) {
    
      // get length of the line
      let len = dist(point1, point2)
    
      // get dot product of the line and circle
      let dot = ( ((circle_center[0]-point1[0])*(point2[0]-point1[0])) + ((circle_center[1]-point1[1])*(point2[1]-point1[1])) ) / Math.pow(len,2);
    
      // find the closest point on the line
      let closestX = point1[0] + (dot * (point2[0]-point1[0]));
      let closestY = point1[1] + (dot * (point2[1]-point1[1]));
    
      // is this point actually on the line segment?
      // if so keep going, but if not, return false
      let onSegment = linePoint(point1, point2, [closestX,closestY]);
      if (!onSegment) return false;
    
      // get distance to closest point
      distX = closestX - circle_center[0];
      distY = closestY - circle_center[1];
      let distance = Math.sqrt( (distX*distX) + (distY*distY) );
    
      if (distance <= circle_radius) {
        return true;
      }
      return false;
    }
   
   function dist(point1, point2) {
      return Math.sqrt(Math.pow(point1[0]-point2[0], 2) + Math.pow(point1[0]-point2[0], 2))
   }
   
   // LINE/POINT
   function linePoint(point1, point2, point3) {
   
      // get distance from the point to the two ends of the line
   
      let d1 = dist(point3, point1);
      let d2 = dist(point3, point2);
    
      // get the length of the line
      let lineLen = dist(point1, point2);
    
      // since floats are so minutely accurate, add
      // a little buffer zone that will give collision
      let buffer = 0.001;    // higher # = less accurate
    
      // if the two distances are equal to the line's
      // length, the point is on the line!
      // note we use the buffer here to give a range,
      // rather than one #
      if (d1+d2 >= lineLen-buffer && d1+d2 <= lineLen+buffer) {
        return true;
      }
      return false;
    }
    return collisions
}


 //console.log(collisions)