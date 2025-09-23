CREATE DATABASE  IF NOT EXISTS `mes_omi` /*!40100 DEFAULT CHARACTER SET utf8 */;
USE `mes_omi`;
-- MySQL dump 10.13  Distrib 5.7.12, for Win32 (AMD64)
--
-- Host: 127.0.0.1    Database: mes_omi
-- ------------------------------------------------------
-- Server version	5.5.5-10.2.7-MariaDB

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `ker_eqp_grp_bt`
--

DROP TABLE IF EXISTS `ker_eqp_grp_bt`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `ker_eqp_grp_bt` (
  `tool_grp_id` varchar(24) NOT NULL,
  `tool_grp` varchar(32) NOT NULL,
  `tool_grp_order` float DEFAULT NULL COMMENT 'd_order: the order show at report\n\ndefault use ope_no, then user can manual turning it\n\nope_no	stage_id\n65.02	Part_Start\n100.00	Compaction\n100.04	Meas_Wgt\n100.08	Meas_Thk\n100.20	Meas_Dust\n200.00	Sintering\n300.00	Sizing\n320.00	Machining\n330.00	Heat_Tx\n400.00	Vibr_Tumbling\n420.00	Plating\n450.00	Steam_Tx\n500.00	Oil_Impg\n600.00	OQC\n600.09	Visual_ispt\n600.20	Density_test\n600.30	Ultrasonic_test\n688.00	Package\n688.88	Part_shipping\n',
  `area_id` varchar(12) DEFAULT NULL,
  `capability` int(11) DEFAULT NULL COMMENT '1. manual set(not use tool peak move)\n2. daily move of eqp_grp',
  `er` varchar(32) DEFAULT NULL COMMENT 'equipment recipe <-- we don''t use logic recipe(it is too complicated)\n2017/06/26 lkchena ',
  `node_id` varchar(16) DEFAULT NULL,
  `node_time` timestamp(6) NULL DEFAULT current_timestamp(6) ON UPDATE current_timestamp(6),
  PRIMARY KEY (`tool_grp_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 AVG_ROW_LENGTH=1820 ROW_FORMAT=DYNAMIC;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `ker_eqp_grp_bt`
--

LOCK TABLES `ker_eqp_grp_bt` WRITE;
/*!40000 ALTER TABLE `ker_eqp_grp_bt` DISABLE KEYS */;
INSERT INTO `ker_eqp_grp_bt` VALUES ('aaa','aaa',88,'aaa',2400,'aaa',NULL,'2020-02-21 05:22:42.883937'),('CMPT-T1','Compaction-20T',100,'C001-A',24000,'20T',NULL,'2020-02-21 05:22:42.883937'),('M-THK-1','MEA-THK',120,'C001-A',30000,'Thinkness-1',NULL,'2020-02-21 05:22:42.883937'),('M-WET-1','MEA-WG',110,'C001-A',30000,'Weight-80',NULL,'2020-02-21 05:22:42.883937'),('SINTER-10','Sintering-1000',230,'S001',24000,'t-1000',NULL,'2020-02-21 05:22:42.883937'),('SINTER-12','Sintering-1200',240,'S001',16000,'t-1200',NULL,'2020-02-21 05:22:42.883937'),('SINTER-3F','Sintering-3F',250,'S003',9600,'*',NULL,'2020-02-21 05:22:42.883937'),('SINTER-6','Sintering-600',210,'S001',20000,'t-600',NULL,'2020-02-21 05:22:42.883937'),('SINTER-8','Sintering-800',220,'S001',24000,'t-800',NULL,'2020-02-21 05:22:42.883937'),('STB','WAFER_START',65,'START-1',24000,'STB',NULL,'2020-02-21 05:22:42.883937');
/*!40000 ALTER TABLE `ker_eqp_grp_bt` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:16:28
