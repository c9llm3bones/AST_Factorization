import com.github.javaparser.*;
import com.github.javaparser.ast.*;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.File;
import java.io.FileWriter;
import java.util.*;

public class ASTBuilder {

    static class Node {
        String type;
        String text;
        List<Node> children = new ArrayList<>();
        Node(String type) { this.type = type; }
    }

    static Node buildNode(com.github.javaparser.ast.Node jpNode) {
        Node node = new Node(jpNode.getClass().getSimpleName());
        if (jpNode.getChildNodes().isEmpty()) {
            node.text = jpNode.toString().replace("\n","\\n");
        } else {
            for (com.github.javaparser.ast.Node c : jpNode.getChildNodes()) {
                node.children.add(buildNode(c));
            }
        }
        return node;
    }

    public static void main(String[] args) throws Exception {
        if(args.length < 2) {
            System.out.println("Usage: java ASTBuilder <input.java> <output.json>");
            return;
        }
        File input = new File(args[0]);
        File output = new File(args[1]);

        CompilationUnit cu = StaticJavaParser.parse(input);
        Node root = buildNode(cu);

        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        try(FileWriter fw = new FileWriter(output)) {
            gson.toJson(root, fw);
        }

        System.out.println("AST saved to " + output.getAbsolutePath());
    }
}
