#target bridge
if (BridgeTalk.appName == "bridge") {
    var menu = MenuElement.create("command", "Rename to DC Identifier (Fixed)", "at the end of thumbnail");
    menu.onSelect = function() {
        renameWithDiagnosticsFixed();
    }
}

function renameWithDiagnosticsFixed() {
    var items = app.document.selections;
    if (items.length == 0) {
        alert("Please select at least one file.");
        return;
    }

    if (ExternalObject.AdobeXMPScript == undefined) {
        ExternalObject.AdobeXMPScript = new ExternalObject("lib:AdobeXMPScript");
    }

    var successCount = 0;
    var failureCount = 0;
    var skipCount = 0;

    for (var i = 0; i < items.length; i++) {
        var thumb = items[i];
        if (thumb.type == "file") {
            try {
                var file = new File(thumb.spec);
                var xmpFile = new XMPFile(file.fsName, XMPConst.UNKNOWN, XMPConst.OPEN_FOR_READ);
                var xmp = xmpFile.getXMP();
                xmpFile.closeFile();

                // Retrieve Dublin Core identifier
                var dcIdentifier = xmp.getProperty(XMPConst.NS_DC, "identifier");
                
                if (dcIdentifier) {
                    // Convert to string and remove leading/trailing spaces via RegEx (safe for older JS)
                    var newNameValue = dcIdentifier.value.toString().replace(/^\s+|\s+$/g, "");
                    
                    if (newNameValue !== "") {
                        // Clean up illegal characters for file systems (\ / : * ? " < > |)
                        newNameValue = newNameValue.replace(/[\\\/:\*\?"<>\|]/g, "_");

                        var extension = file.name.substring(file.name.lastIndexOf("."));
                        var newFullName = newNameValue + extension;

                        // Rename the file
                        var success = file.rename(newFullName);
                        if (success) {
                            successCount++;
                        } else {
                            failureCount++;
                        }
                    } else {
                        skipCount++; // Found property but it was blank
                    }
                } else {
                    skipCount++; // No dc:identifier found
                }

            } catch (e) {
                failureCount++;
            }
        }
    }

    // Refresh view and report
    app.document.chooseMenuItem("Refresh");
    alert("Process Complete.\n\nSuccessfully Renamed: " + successCount + " files.\nSkipped (No ID): " + skipCount + " files.\nFailed (System/Permissions): " + failureCount + " files.");
}