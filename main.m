% Load data and header files
dataFileSegmented = 'tree_segmented.dat';
headerFileSegmented = 'tree_segmented.hdr';
originalDataFile = 'taglio_Ostuni_EL_gorgognolo.img';
originalHeaderFile = 'taglio_Ostuni_EL_gorgognolo.hdr';
thermalTermFile = 'Reg_Term_Ostuni_fc4.tif';
kmlFile = 'doc.kml';

% Set random number generator seed
rng(42)

% Read georeferenced images
hypercubeOriginal = hypercube(originalDataFile, originalHeaderFile);
image415 = hypercubeOriginal.DataCube(:, :, 5);
image435 = hypercubeOriginal.DataCube(:, :, 9);
[imgX, imgY] = size(image435);

% Calculate NPQI Image
npqiImage = (image415 - image435) ./ (image415 + image435);

% Calculate NDVI Image
ndviImage = ndvi(hypercubeOriginal);

% Read georeferenced images and configure registration
[segmentedImage, RSegmented] = readgeoraster(dataFileSegmented);
[thermalImage, RTerm] = readgeoraster(thermalTermFile);

% Rescale image values
for i = 1:47
    segmentedImage(:, :, i) = rescale(segmentedImage(:, :, i));
end

% Select the segmented canopy level
canopyLayer = segmentedImage(:, :, 47);
RA = imref2d(size(canopyLayer));
RB = imref2d(size(thermalImage));

% Configure image registration
[optimizer, metric] = imregconfig("multimodal");
registeredThermalImage = imregister(thermalImage, RB, canopyLayer, RA, 'translation', optimizer, metric);

% Extract regions from binary mask
regions = logical(canopyLayer);
boundingBoxes = regionprops(regions, 'BoundingBox');

% Read KML data and project it to the segmented image CRS
kmlData = kml2struct(kmlFile);
proj = RSegmented.ProjectedCRS;
treeData = struct2table(kmlData);
treeData.Geometry = [];
treeData.Description = [];
[treeData.geoX, treeData.geoY] = projfwd(proj, treeData.Lat, treeData.Lon);
[treeData.xIntrinsic, treeData.yIntrinsic] = worldToIntrinsic(RSegmented, treeData.geoX, treeData.geoY);

% Offset points
treeData.xIntrinsic = treeData.xIntrinsic - 2;
treeData.yIntrinsic = treeData.yIntrinsic + 4;

% Remove undetermined tree indices
indicesToRemove = [77, 120, 161];
treeData(indicesToRemove, :) = [];

% Calculate coordinate limits on images
xMax = max(treeData.xIntrinsic);
xMin = min(treeData.xIntrinsic);
yMax = max(treeData.yIntrinsic);
yMin = min(treeData.yIntrinsic);

% Filter bounding boxes based on specified criteria
filteredBoundingBoxes = [];
for i = 1:numel(boundingBoxes)
    if boundingBoxes(i).BoundingBox(1) >= xMin - 20 && boundingBoxes(i).BoundingBox(1) <= xMax + 20 && boundingBoxes(i).BoundingBox(2) >= yMin - 20 && boundingBoxes(i).BoundingBox(2) <= yMax + 20 && boundingBoxes(i).BoundingBox(3) < 60 && boundingBoxes(i).BoundingBox(4) < 60
        filteredBoundingBoxes = [filteredBoundingBoxes; boundingBoxes(i)];
    end
end

% Initialize flags for tracking assigned bounding boxes
isBoundingBoxAssigned = false(numel(filteredBoundingBoxes), 1);
assignedBoundingBoxes = [];

% Associate bounding boxes with trees
for t = 1:size(treeData, 1)
    distances = zeros(1, numel(filteredBoundingBoxes));
    
    for i = 1:numel(filteredBoundingBoxes)
        if ~isBoundingBoxAssigned(i)
            bb = filteredBoundingBoxes(i);
            bbCenter = [bb.BoundingBox(1) + (bb.BoundingBox(3) / 2), bb.BoundingBox(2) + (bb.BoundingBox(4) / 2)];
            distances(i) = sqrt((treeData.xIntrinsic(t) - bbCenter(1)).^2 + (treeData.yIntrinsic(t) - bbCenter(2)).^2);
        else
            distances(i) = inf;
        end
    end
    
    filteredDistances = distances;

    [~, minIdx] = min(filteredDistances);
    isBoundingBoxAssigned(minIdx) = true;
    assignedBoundingBoxes = [assignedBoundingBoxes, filteredBoundingBoxes(minIdx)];
end
treeData.BoundingBox = assignedBoundingBoxes';

% Display segmented images and bounding boxes
numTrees = size(treeData, 1);
colorMap = rand([numTrees, 3]);
figure;
imshowpair(canopyLayer, registeredThermalImage, "blend")
hold on;
for i = 1:numTrees
    tempBoundingBox = treeData(i, :).BoundingBox;
    color = colorMap(i, :);
    rectangle('Position', tempBoundingBox.BoundingBox, 'EdgeColor', color, 'LineWidth', 2);
end
scatter(treeData.xIntrinsic, treeData.yIntrinsic, 50, colorMap(1:numTrees, :), 'filled');
text(treeData.xIntrinsic, treeData.yIntrinsic, treeData.Name, 'FontSize', 8, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');

hold off;

% Calculate features from bounding box regions
spectralFeatures = zeros(numTrees, 47);
thermalFeatures = zeros(numTrees, size(registeredThermalImage, 3));
originalFeatures = zeros(numTrees, 47);

for t = 1:numTrees
    x = round(treeData.BoundingBox(t).BoundingBox(1));
    y = round(treeData.BoundingBox(t).BoundingBox(2));
    dimX = round(treeData.BoundingBox(t).BoundingBox(3));
    dimY = round(treeData.BoundingBox(t).BoundingBox(4));
    spectralROI = segmentedImage(y:y+dimY, x:x+dimX, :);
    originalSpectralROI = hypercubeOriginal.DataCube(y:y+dimY, x:x+dimX, :);
    thermalROI = registeredThermalImage(y:y+dimY, x:x+dimX, :);
    npqiROI = npqiImage(y:y+dimY, x:x+dimX, :);
    ndviROI = ndviImage(y:y+dimY, x:x+dimX, :);
    [dimSpecX, dimSpecY, ~] = size(spectralROI);
    spectralSum = 0;
    originalSum = 0;
    npqiSum = 0;
    ndviSum = 0;
    numSum = 0;
    thermalSum = 0;
    
    for j = 1:dimSpecX
        for i = 1:dimSpecY
            if spectralROI(j, i) > 0
                spectralSum = spectralSum + spectralROI(j, i, :);
                originalSum = originalSum + originalSpectralROI(j, i, :);
                thermalSum = thermalSum + thermalROI(j, i, :);
                ndviSum = ndviSum + ndviROI(j, i, :);
                npqiSum = npqiSum + npqiROI(j, i, :);
                numSum = numSum + 1;
            end
        end
    end
    spectralFeatures(t, :) = spectralSum / numSum;
    originalFeatures(t, :) = originalSum / numSum;
    thermalFeatures(t, :) = thermalSum / numSum;
    ndviFeatures(t, :) = ndviSum / numSum;
    npqiFeatures(t, :) = npqiSum / numSum;
    latFeatures(t) = treeData.Lat(t);
    lonFeatures(t) = treeData.Lon(t);
end

% Rescale latitude and longitude features
latFeatures = rescale(latFeatures, 0, 1);
lonFeatures = rescale(lonFeatures, 0, 1);

% Create feature set
originalX = [originalFeatures, ndviFeatures, npqiFeatures, thermalFeatures];
originalT = treeData.GT;

% Save features in csv file
writematrix(originalX, 'originalFeatures.csv');
writematrix(originalT, 'originalTargets.csv');