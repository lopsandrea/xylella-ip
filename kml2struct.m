function kmlStruct = kml2struct(kmlFile)
    % A function to parse KML data and convert it into a structured format.
    
    % Open the KML file for reading
    [FID, msg] = fopen(kmlFile, 'rt');
    
    % Check if the file was successfully opened
    if FID < 0
        error(msg);
    end
    
    % Read the contents of the KML file
    txt = fread(FID, 'uint8=>char')';
    fclose(FID);
    
    % Regular expression to match Placemark elements in the KML
    expr = '<Placemark.+?>.+?</Placemark>';
    
    % Extract all Placemark elements as strings
    objectStrings = regexp(txt, expr, 'match');
    
    % Get the total number of Placemark elements
    Nos = length(objectStrings);
    
    % Initialize an empty structure array to store parsed KML data
    kmlStruct = struct('Geometry', {}, 'Name', {}, 'Description', {}, 'Lon', {}, 'Lat', {}, 'GT', {}, 'BoundingBox', {});
    
    % Loop through each Placemark element and parse its data
    for ii = 1:Nos
        % Find the Object Name Field within the Placemark element
        nameBucket = regexp(objectStrings{ii}, '<name.*?>.+?</name>', 'match');
        if isempty(nameBucket)
            name = 'undefined';
        else
            % Remove XML tags and whitespace to extract the name
            name = regexprep(nameBucket{1}, '<name.*?>\s*', '');
            name = regexprep(name, '\s*</name>', '');
        end

        % Find the Object GT Field within the Placemark element
        gtBucket = regexp(objectStrings{ii}, '<SimpleData name="GT".*?>.+?</SimpleData>', 'match');
        if isempty(gtBucket)
            gt = 'undefined';
        else
            % Remove XML tags and whitespace to extract the GT value
            gt = regexprep(gtBucket{1}, '<SimpleData name="GT".*?>\s*', '');
            gt = regexprep(gt, '\s*</SimpleData>', '');
            gt = str2double(gt);
        end

        % Find the Object Description Field within the Placemark element
        descBucket = regexp(objectStrings{ii}, '<description.*?>.+?</description>', 'match');
        if isempty(descBucket)
            desc = '';
        else
            % Remove XML tags and whitespace to extract the description
            desc = regexprep(descBucket{1}, '<description.*?>\s*', '');
            desc = regexprep(desc, '\s*</description>', '');
        end

        % Identify the type of geometry (Point, Line, or Polygon)
        geom = 0;
        if ~isempty(regexp(objectStrings{ii}, '<Point', 'once'))
            geom = 1;
        elseif ~isempty(regexp(objectStrings{ii}, '<LineString', 'once'))
            geom = 2;
        elseif ~isempty(regexp(objectStrings{ii}, '<Polygon', 'once'))
            geom = 3;
        end

        % Map the numeric geometry code to a human-readable string
        switch geom
            case 1
                geometry = 'Point';
            case 2
                geometry = 'Line';
            case 3
                geometry = 'Polygon';
            otherwise
                geometry = '';
        end

        % Find the Coordinate Field within the Placemark element
        coordBucket = regexp(objectStrings{ii}, '<coordinates.*?>.+?</coordinates>', 'match');
        % Remove XML tags and whitespace to extract the coordinate string
        coordStr = regexprep(coordBucket{1}, '<coordinates.*?>(\s+)*', '');
        coordStr = regexprep(coordStr, '(\s+)*</coordinates>', '');
        % Split the coordinate string by commas or white spaces and convert to doubles
        coordMat = str2double(regexp(coordStr, '[,\s]+', 'split'));
        % Rearrange coordinates into an x-by-3 matrix
        [m, n] = size(coordMat);
        coordMat = reshape(coordMat, 3, m * n / 3)';

        % Define polygons in clockwise direction and terminate
        [Lat, Lon] = poly2ccw(coordMat(:, 2), coordMat(:, 1));
        if geom == 3
            Lon = [Lon; NaN];
            Lat = [Lat; NaN];
        end

        % Calculate the bounding box of the geometry
        boundingBox = [[min(Lon), min(Lat); max(Lon), max(Lat)]];

        % Create a structure entry for the current Placemark
        kmlStruct(ii).Geometry = geometry;
        kmlStruct(ii).Name = name;
        kmlStruct(ii).Description = desc;
        kmlStruct(ii).Lon = Lon;
        kmlStruct(ii).Lat = Lat;
        kmlStruct(ii).GT = gt;
        kmlStruct(ii).BoundingBox = boundingBox;
    end
end
