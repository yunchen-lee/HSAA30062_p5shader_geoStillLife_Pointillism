var program1;
var buffer1;

function preload() {
    program1 = loadShader('shaders/geoStillLife.vert', 'shaders/geoStillLife.frag');
}

function setup() {
    createCanvas(windowWidth, windowHeight, WEBGL);
    // createCanvas(windowWidth, windowHeight, WEBGL);

    noStroke();

    buffer1 = createGraphics(width, height, WEBGL);
    buffer1.noStroke();

    pixelDensity(1);


}

function pass1(outputBuffer, sh) {
    outputBuffer.shader(sh);
    sh.setUniform("u_resolution", [width * 2, height * 2]);
    // sh.setUniform("u_resolution", [width, height]);
    sh.setUniform('u_mouse', [mouseX, mouseY]);
    sh.setUniform('u_time', millis() / 1000.0);
    outputBuffer.rect(-width / 2, -height / 2, width, height);
}

function draw() {
    pass1(
        buffer1,
        program1
    );

    image(buffer1, -width / 2, -height / 2, width, height);

}

function mouseClicked() {
    save('myCanvas.png');
}