% Load data and header files
datafile = 'tree_segmented.dat';
hdrfile = 'tree_segmented.hdr';
originaldatafile = 'taglio_Ostuni_EL_gorgognolo.img';
originalhdrfile = 'taglio_Ostuni_EL_gorgognolo.hdr';
termfile = 'Reg_Term_Ostuni_fc4.tif';
kmlFile = 'doc.kml';

% Set random number generator seed
rng(42)

% Read georeferenced images
hcube = hypercube(originaldatafile, originalhdrfile);
image415 = hcube.DataCube(:, :, 5);
image435 = hcube.DataCube(:, :, 9);
[dimImgX, dimImgY] = size(image435);

% Calculate NPQI Image
NPQIImage = zeros(dimImgX, dimImgY);
for i = 1:dimImgX
    for j = 1:dimImgY
        NPQIImage(i, j) = (image415(i, j) - image435(i, j)) / (image415(i, j) + image435(i, j));
    end
end

% Calculate NDVI Image
NDVIImage = ndvi(hcube);

% Read georeferenced images and configure registration
[originalsegmentedImage, originalRSegmented] = readgeoraster(originaldatafile);
[segmentedImage, RSegmented] = readgeoraster(datafile);
[termImage, RTerm] = readgeoraster(termfile);

% Rescale image values
for i = 1:47
    originalsegmentedImage(:, :, i) = rescale(originalsegmentedImage(:, :, i));
    segmentedImage(:, :, i) = rescale(segmentedImage(:, :, i));
end

% Select the segmented canopy level
bwSegmentedImage = segmentedImage(:, :, 47);
RA = imref2d(size(bwSegmentedImage));
RB = imref2d(size(termImage));

% Configure image registration
[optimizer, metric] = imregconfig("multimodal");
termImageRegistered = imregister(termImage, RB, bwSegmentedImage, RA, 'translation', optimizer, metric);

% Extract regions from binary mask
regions = logical(bwSegmentedImage);
boundingBoxes = regionprops(regions, 'BoundingBox');

% Read KML data and project it to the segmented image CRS
kmlData = kml2struct(kmlFile);
proj = RSegmented.ProjectedCRS;
trees = struct2table(kmlData);
trees.Geometry = [];
trees.Description = [];
[trees.geoX, trees.geoY] = projfwd(proj, trees.Lat, trees.Lon);
[trees.xIntrinsic, trees.yIntrinsic] = worldToIntrinsic(RSegmented, trees.geoX, trees.geoY);

% Offset points
trees.xIntrinsic = trees.xIntrinsic - 2;
trees.yIntrinsic = trees.yIntrinsic + 4;

% Remove undetermined tree indices
indicesToRemove = [77, 120, 161];
trees(indicesToRemove, :) = [];

% Calculate coordinate limits on images
xMax = max(trees.xIntrinsic);
xMin = min(trees.xIntrinsic);
yMax = max(trees.yIntrinsic);
yMin = min(trees.yIntrinsic);

% Filter bounding boxes based on specified criteria
nuovoBoundingBoxes = [];
for i = 1:numel(boundingBoxes)
    if boundingBoxes(i).BoundingBox(1) >= xMin - 20 && boundingBoxes(i).BoundingBox(1) <= xMax + 20 && ...
       boundingBoxes(i).BoundingBox(2) >= yMin - 20 && boundingBoxes(i).BoundingBox(2) <= yMax + 20 && ...
       boundingBoxes(i).BoundingBox(3) < 60 && boundingBoxes(i).BoundingBox(4) < 60
        nuovoBoundingBoxes = [nuovoBoundingBoxes; boundingBoxes(i)];
    end
end

% Initialize flags for tracking assigned bounding boxes
bbAssigned = false(numel(nuovoBoundingBoxes), 1);
BB = [];

% Associate bounding boxes with trees
for t = 1:size(trees, 1)
    distances = zeros(1, numel(nuovoBoundingBoxes));
    
    for i = 1:numel(nuovoBoundingBoxes)
        if ~bbAssigned(i)
            box = nuovoBoundingBoxes(i);
            box_center = [box.BoundingBox(1) + (box.BoundingBox(3) / 2), box.BoundingBox(2) + (box.BoundingBox(4) / 2)];
            distances(i) = sqrt((trees.xIntrinsic(t) - box_center(1)).^2 + (trees.yIntrinsic(t) - box_center(2)).^2);
        else
            distances(i) = inf;
        end
    end
    
    filteredDistances = distances;

    [~, min_idx] = min(filteredDistances);
    bbAssigned(min_idx) = true;
    BB = [BB, nuovoBoundingBoxes(min_idx)];
end
trees.BB = BB';

% Display segmented images and bounding boxes
numTrees = size(trees, 1);
colorMap = rand([numTrees, 3]);
figure;
imshowpair(bwSegmentedImage, termImageRegistered, "blend")
hold on;
for i = 1:numTrees
    tempbb = trees(i, :);
    boundingBox = tempbb.BB.BoundingBox;
    colore = colorMap(i, :);
    rectangle('Position', boundingBox, 'EdgeColor', colore, 'LineWidth', 2);
end
scatter(trees.xIntrinsic, trees.yIntrinsic, 50, colorMap(1:numTrees, :), 'filled');
text(trees.xIntrinsic, trees.yIntrinsic, trees.Name, 'FontSize', 8, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');

hold off;

% Calculate features from bounding box regions
averSpectralFeatures = zeros(numTrees, 47);
averThermalFeatures = zeros(numTrees, size(termImageRegistered, 3));
averOriginalFeatures = zeros(numTrees, 47);

for t = 1:numTrees
    x = round(trees.BB(t).BoundingBox(1));
    y = round(trees.BB(t).BoundingBox(2));
    dimX = round(trees.BB(t).BoundingBox(3));
    dimY = round(trees.BB(t).BoundingBox(4));
    spectralROI = segmentedImage(y:y+dimY, x:x+dimX, :);
    originalspectralROI = originalsegmentedImage(y:y+dimY, x:x+dimX, :);
    thermalROI = termImageRegistered(y:y+dimY, x:x+dimX, :);
    npqiROI = NPQIImage(y:y+dimY, x:x+dimX, :);
    ndviROI = NDVIImage(y:y+dimY, x:x+dimX, :);
    [dimSpecX, dimSpecY, ~] = size(spectralROI);
    tempSpectralSum = 0;
    tempOriginalSum = 0;
    tempnpqi = 0;
    tempndvi = 0;
    numTempSum = 0;
    tempThermSum = 0;
    for j = 1:dimSpecX
        for i = 1:dimSpecY
            if spectralROI(j, i) > 0
                tempSpectralSum = tempSpectralSum + spectralROI(j, i, :);
                tempOriginalSum = tempOriginalSum + originalspectralROI(j, i, :);
                tempThermSum = tempThermSum + thermalROI(j, i, :);
                tempndvi = tempndvi + ndviROI(j, i, :);
                tempnpqi = tempnpqi + npqiROI(j, i, :);
                numTempSum = numTempSum + 1;
            end
        end
    end
    averSpectralFeatures(t, :) = tempSpectralSum / numTempSum;
    averOriginalFeatures(t, :) = tempOriginalSum / numTempSum;
    averThermalFeatures(t, :) = tempThermSum / numTempSum;
    averndviFeatures(t, :) = tempndvi / numTempSum;
    avernpqiFeatures(t, :) = tempnpqi / numTempSum;
    latFeatures(t) = trees.Lat(t);
    lonFeatures(t) = trees.Lon(t);
end

% Rescale latitude and longitude features
latFeatures = rescale(latFeatures, 0, 1);
lonFeatures = rescale(lonFeatures, 0, 1);

% Create feature set
x = [averOriginalFeatures, averndviFeatures, avernpqiFeatures, averThermalFeatures];
t = trees.GT;
